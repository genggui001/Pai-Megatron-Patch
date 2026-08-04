from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from fused_kl_loss import kl_ce_loss

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_virtual_pipeline_model_parallel_world_size,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint


def load_teacher_checkpoint(
    teacher_model: nn.Module,
    load_arg: str = "base_model_load",
    strict: bool = True,
    ckpt_format: Optional[str] = None,
) -> None:
    """Load the frozen teacher checkpoint without changing student resume flow."""

    args = get_args()
    original_args_finetune = args.finetune
    original_ckpt_format = args.ckpt_format
    args.finetune = True
    if ckpt_format is not None:
        args.ckpt_format = ckpt_format

    try:
        print_rank_0(f"Loading base model from {getattr(args, load_arg)} ...")
        load_checkpoint([teacher_model], None, None, load_arg=load_arg, strict=strict)
        print_rank_0("...base model loaded successfully.")
    finally:
        args.finetune = original_args_finetune
        args.ckpt_format = original_ckpt_format


class KLDistillationWrapper(nn.Module):
    """Wrap student/teacher models with ModelOpt-compatible PP packing."""

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        kl_loss_weight: float,
        kl_reverse: bool = False,
    ) -> None:
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.kl_loss_weight = kl_loss_weight
        self.kl_reverse = kl_reverse
        self._config = get_attr_wrapped_model(student_model, "config", allow_none=False)
        self._tensor_split_idx = self._config.hidden_size
        self._freeze_teacher_model()
        print_rank_0("KLDistillationWrapper using fused_kl_loss.")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            student_model = self._modules.get("student_model")
            if student_model is not None:
                return getattr(student_model, name)
            raise exc

    def _freeze_teacher_model(self) -> None:
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.student_model.train(mode)
        self.teacher_model.eval()
        return self

    def sharded_state_dict(self, *args, **kwargs):
        return self.student_model.sharded_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.student_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.student_model.load_state_dict(state_dict, strict=strict)

    def set_student_input_tensor_shape(self, shapes: List[Tuple[int, ...]]) -> None:
        if shapes and shapes[0] is not None:
            self._tensor_split_idx = shapes[0][-1]

    def set_input_tensor(self, input_tensors: Union[List[Tensor], Tensor]) -> None:
        if not isinstance(input_tensors, list):
            input_tensors = [input_tensors]

        if (
            not self.training
            or get_pipeline_model_parallel_world_size() == 1
            or input_tensors[0] is None
        ):
            self.student_model.set_input_tensor(input_tensors)
            self.teacher_model.set_input_tensor(input_tensors)
            return

        teacher_inputs = [
            tensor[..., self._tensor_split_idx :] if tensor is not None else tensor
            for tensor in input_tensors
        ]
        student_inputs = [
            tensor[..., : self._tensor_split_idx] if tensor is not None else tensor
            for tensor in input_tensors
        ]
        self.student_model.set_input_tensor(student_inputs)
        self.teacher_model.set_input_tensor(teacher_inputs)

    def get_tensor_shapes_adjust_fn(
        self,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
    ) -> Optional[Callable]:
        if (
            forward_only
            or get_pipeline_model_parallel_world_size() == 1
            or get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            return None

        teacher_config = get_attr_wrapped_model(
            self.teacher_model, "config", allow_none=False
        )
        student_pg_collection = get_attr_wrapped_model(
            self.student_model, "pg_collection", allow_none=False
        )

        def adjust_tensor_shapes(
            recv_tensor_shapes: List[Tuple[int, ...]],
            send_tensor_shapes: List[Tuple[int, ...]],
        ):
            teacher_tensor_shapes = get_tensor_shapes(
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                decoder_seq_length=decoder_seq_length,
                config=teacher_config,
                tp_group=student_pg_collection.tp,
                cp_group=student_pg_collection.cp,
            )
            teacher_hidden_size = teacher_tensor_shapes[0][-1]
            self.set_student_input_tensor_shape(recv_tensor_shapes)

            for idx, shape in enumerate(recv_tensor_shapes):
                if shape is None:
                    continue
                recv_tensor_shapes[idx] = (*shape[:-1], shape[-1] + teacher_hidden_size)

            for idx, shape in enumerate(send_tensor_shapes):
                if shape is None:
                    continue
                send_tensor_shapes[idx] = (*shape[:-1], shape[-1] + teacher_hidden_size)

            return recv_tensor_shapes, send_tensor_shapes

        return adjust_tensor_shapes

    def _forward_last_stage_with_fused_kl(self, *args, **kwargs):
        student_kwargs = dict(kwargs)
        teacher_kwargs = dict(kwargs)

        student_kwargs["return_hidden_states"] = True
        student_h, student_weight = self.student_model(*args, **student_kwargs)

        teacher_kwargs["labels"] = None
        teacher_kwargs["return_hidden_states"] = True
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_h, teacher_weight = self.teacher_model(*args, **teacher_kwargs)
            teacher_h = teacher_h.detach()
            teacher_weight = teacher_weight.detach()

        labels = kwargs["labels"]
        S, B, H = student_h.shape
        student_h_2d = student_h.reshape(-1, H).contiguous()
        teacher_h_2d = teacher_h.reshape(-1, H).contiguous()
        labels_1d = labels.reshape(-1).contiguous()

        tp_group = (
            get_tensor_model_parallel_group()
            if self._config.tensor_model_parallel_size > 1
            else None
        )
        kl_weight_t = torch.tensor(
            [self.kl_loss_weight], device=student_h.device, dtype=torch.float32,
        )

        result = kl_ce_loss(
            student_h_2d, teacher_h_2d,
            student_weight, teacher_weight,
            labels_1d, kl_weight_t,
            kl_reverse=self.kl_reverse,
            tp_group=tp_group,
        )

        return {
            "all_loss": result.loss.reshape(S, B),
            "ce_loss": result.ce_loss.reshape(S, B),
            "kl_loss": result.kl_loss.reshape(S, B),
            "kl_loss_weight": self.kl_loss_weight,
        }

    def forward(self, *args, **kwargs):
        if not self.training:
            return self.student_model(*args, **kwargs)

        teacher_kwargs = dict(kwargs)
        student_kwargs = dict(kwargs)

        if not is_pipeline_last_stage():
            teacher_kwargs["labels"] = None
            with torch.no_grad():
                self.teacher_model.eval()
                teacher_output = self.teacher_model(*args, **teacher_kwargs)
                teacher_output = teacher_output.detach()
            student_output = self.student_model(*args, **student_kwargs)
            return torch.cat([student_output, teacher_output], dim=-1)

        return self._forward_last_stage_with_fused_kl(*args, **kwargs)
