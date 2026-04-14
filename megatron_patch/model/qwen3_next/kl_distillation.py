from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
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

from megatron_patch.model.qwen3_next.triton_grouped_kl import (
    HAS_TRITON,
    triton_grouped_kl_tp,
)


class _AllReduce(torch.autograd.Function):
    """All-reduce that preserves backward propagation."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(
    tensor: Tensor,
    op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
) -> Tensor:
    """All-reduce helper for TP-aware logits KL."""

    return _AllReduce.apply(op, group, tensor)


class TensorParallelLogitsKLLoss(nn.Module):
    """KL loss that matches ModelOpt's tensor-parallel math.

    Convention used here:
    - student/predictions = current finetuned model = Q
    - teacher/targets = frozen base/reference model = P
    - reverse=False => forward KL in the paper notation: KL(P || Q)
    - reverse=True => reverse KL in the paper notation: KL(Q || P)
    """

    def __init__(self, model_config, temperature: float = 1.0, reverse: bool = False):
        super().__init__()
        self._config = model_config
        self._temperature = temperature
        self._reverse = reverse

    def _torch_forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        targets = targets.detach()

        # In this file, predictions are student logits and targets are teacher logits.
        output_teacher = targets.float()
        output_teacher.div_(self._temperature)
        output_student = predictions.float()
        output_student.div_(self._temperature)

        if self._config.tensor_model_parallel_size > 1:
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_teacher.sub_(teacher_logits_max.unsqueeze(dim=-1))

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            torch.distributed.all_reduce(denom_teacher, group=get_tensor_model_parallel_group())

            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_student.sub_(student_logits_max.unsqueeze(dim=-1).detach())

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(
                denom_student, group=get_tensor_model_parallel_group()
            )
        else:
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            output_teacher.sub_(teacher_logits_max.unsqueeze(dim=-1))

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)

            student_logits_max, _ = torch.max(output_student, dim=-1)
            output_student.sub_(student_logits_max.unsqueeze(dim=-1).detach())

            denom_student = torch.sum(torch.exp(output_student), dim=-1)

        # Convert centered logits to log-probabilities in place to avoid allocating
        # extra full-size student_log_prob / teacher_log_prob tensors.
        denom_teacher.log_()
        output_teacher.sub_(denom_teacher.unsqueeze(dim=-1))

        denom_student.log_()
        output_student.sub_(denom_student.unsqueeze(dim=-1))

        if self._reverse:
            # torch.nn.functional.kl_div(input, target, log_target=True)
            # computes KL(target || input), so this branch is KL(student || teacher).
            loss = torch.sum(
                F.kl_div(
                    output_teacher,
                    output_student,
                    reduction="none",
                    log_target=True,
                ),
                dim=-1,
            )
        else:
            # Default branch: KL(teacher || student), which is the "forward KL"
            # under the paper's notation where P=teacher and Q=student.
            loss = torch.sum(
                F.kl_div(
                    output_student,
                    output_teacher,
                    reduction="none",
                    log_target=True,
                ),
                dim=-1,
            )
        return loss

    def forward(self, predictions: Tensor, targets: Tensor):
        tp_group = (
            get_tensor_model_parallel_group()
            if self._config.tensor_model_parallel_size > 1
            else None
        )

        if HAS_TRITON and predictions.is_cuda and targets.is_cuda:
            loss = triton_grouped_kl_tp(
                predictions,
                targets.detach(),
                temperature=self._temperature,
                reverse=self._reverse,
                tp_group=tp_group,
            )
        else:
            loss = self._torch_forward(predictions, targets)

        if tp_group is not None:
            loss = tensor_parallel.reduce_from_tensor_model_parallel_region(loss)

        return (loss.transpose(0, 1).contiguous(), False, False)


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
        kl_temperature: float = 1.0,
        kl_reverse: bool = False,
    ) -> None:
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.kl_loss_weight = kl_loss_weight
        self.kl_loss = TensorParallelLogitsKLLoss(
            get_attr_wrapped_model(student_model, "config", allow_none=False),
            temperature=kl_temperature,
            reverse=kl_reverse,
        )
        self._tensor_split_idx = get_attr_wrapped_model(
            student_model, "config", allow_none=False
        ).hidden_size
        self._freeze_teacher_model()

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

        student_kwargs["return_raw_logits"] = True
        student_loss, student_logits = self.student_model(*args, **student_kwargs)

        teacher_kwargs["labels"] = None
        teacher_kwargs["return_raw_logits"] = True
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_logits = self.teacher_model(*args, **teacher_kwargs)
            teacher_logits = teacher_logits.detach()

        return {
            "ce_loss": student_loss,
            "kl_loss": self.kl_loss(student_logits, teacher_logits),
            "kl_loss_weight": self.kl_loss_weight,
        }
