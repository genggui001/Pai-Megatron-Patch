# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import NoReturn, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import (
    deprecate_inference_params,
    is_fa_min_version,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from einops import rearrange
except ImportError:
    rearrange = None


try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
    )
except ImportError:
    HAVE_TE = False
    SplitAlongDim, TELinear, set_save_original_input = None, None, None

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    from flashattn_hopper.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )

    HAVE_FA3 = True
except:
    HAVE_FA3 = False

class GatedSoftmaxAttention(Attention):
    """Gated softmax attention layer class

    Gated softmax attention layer takes input with size [s, b, h]
    and returns output of the same size.

    The linear_qkv in `SelfAttentionSubmodules` is used for creating `linear_qkvg`
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.fake_qkv_quantization_type = getattr(config, 'attention_fake_qkv_quantization_type', None)
        if self.fake_qkv_quantization_type == "fp8_e4m3":
            from qdq import fake_fp8_e4m3_quantization_ste
            self._fake_qkv_ste = fake_fp8_e4m3_quantization_ste
            self.register_buffer(
                "_qkv_query_amax_calibration", torch.zeros(1), persistent=False
            )
            self.register_buffer(
                "_qkv_key_amax_calibration", torch.zeros(1), persistent=False
            )
            self.register_buffer(
                "_qkv_value_amax_calibration", torch.zeros(1), persistent=False
            )

        self.linear_qgkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            2 * (self.query_projection_size + self.kv_projection_size),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during
        data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably
        not required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())

        def _compare(srcs, tgts, names, parallelism):
            assert len(srcs) == len(tgts) == len(names)
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(src == tgt), (
                    f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. "
                    f"Diff: {torch.norm(src - tgt)}"
                )

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * 2 * (np/ng + 1) * hn)]
        mixed_qgkv, _ = self.linear_qgkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, 2 * (np/ng + 1) * hn]
        new_tensor_shape = mixed_qgkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                2 * (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 1)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qgkv = mixed_qgkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, gate, key, value) = SplitAlongDim(mixed_qgkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, gate, key, value) = torch.split(mixed_qgkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        gate = gate.reshape(query.size(0), query.size(1), -1)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, gate, key, value

    def _qdq_flash_varlen_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        value_scale: Tensor,
        attn_mask_type: AttnMaskType,
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tensor:
        """Packed BF16 attention with FP8 QDQ inside flash_attn_varlen_func_fp8_qdq."""
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError(
                "--attention-backend qdq_flash_attn requires packed sequences (qkv_format=thd)."
            )
        if self.config.context_parallel_size != 1:
            raise ValueError(
                "--attention-backend qdq_flash_attn requires context parallel size 1."
            )
        if attn_mask_type not in (
            AttnMaskType.causal,
            AttnMaskType.padding_causal,
            AttnMaskType.causal_bottom_right,
        ):
            raise ValueError(
                "--attention-backend qdq_flash_attn only supports causal masks, "
                f"got {attn_mask_type}."
            )
        if query.dtype != torch.bfloat16:
            raise ValueError(
                "--attention-backend qdq_flash_attn requires BF16 QKV, "
                f"got {query.dtype}."
            )

        from qdq.flash_attn import flash_attn_varlen_func_fp8_qdq

        def _python_int(value) -> int:
            if isinstance(value, torch.Tensor):
                return int(value.item())
            return int(value)

        window_size = self.config.window_size if self.config.window_size is not None else (-1, -1)
        return flash_attn_varlen_func_fp8_qdq(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            packed_seq_params.cu_seqlens_q.to(dtype=torch.int32).contiguous(),
            packed_seq_params.cu_seqlens_kv.to(dtype=torch.int32).contiguous(),
            _python_int(packed_seq_params.max_seqlen_q),
            _python_int(packed_seq_params.max_seqlen_kv),
            dropout_p=self.config.attention_dropout,
            softmax_scale=self.config.softmax_scale,
            causal=True,
            window_size=tuple(window_size),
            deterministic=self.config.deterministic_mode,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            qdq_mode="fa3",
        )

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations"""
        self._backward_qkv_proj()
        self._backward_output_proj()

    def _backward_qkv_proj(self):
        """Update weights for QKV projection layer"""
        self.linear_qkv.backward_dw()

    def _backward_output_proj(self):
        """Update weights for output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        from megatron.core.extensions.transformer_engine import set_save_original_input

        set_save_original_input(self.linear_qkv)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                "2.7.3"
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qgkv")
        query, gate, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        nvtx_range_pop(suffix="qgkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
        )

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            raise ValueError("Flash Decoding of Gated Softmax Attention is not supported!")


        if (
            in_decode_mode
            and self.config.enable_cuda_graph
            and self.config.cuda_graph_scope != "full_iteration"
            and inference_context.is_static_batching()
        ):
            raise ValueError(f"CUDA graphs must use flash decode with static batching!")

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                rotary_pos_cos_sin,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        cp_group=self.pg_collection.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.pg_collection.cp,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # QKV fake quantization (before core attention)
        # Scale synchronized across TP ranks for consistency with vLLM inference
        # scale = global_amax / FP8_E4M3_MAX, clamped to avoid zero
        # qdq_flash_attn consumes these scales inside the kernel and skips STE.
        # ==================================
        use_qdq_flash_attn = self.config.attention_backend == AttnBackend.qdq_flash_attn
        query_scale = key_scale = value_scale = None
        if self.fake_qkv_quantization_type is not None or use_qdq_flash_attn:
            if use_qdq_flash_attn and self.fake_qkv_quantization_type != "fp8_e4m3":
                raise ValueError(
                    "--attention-backend qdq_flash_attn requires "
                    "--attention-fake-qkv-quantization-type fp8_e4m3."
                )
            _FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
            _TINY = torch.finfo(torch.float32).tiny

            query_amax = query.detach().abs().amax().float()
            key_amax = key.detach().abs().amax().float()
            value_amax = value.detach().abs().amax().float()
            tp_group = self.pg_collection.tp
            if tp_group is not None:
                torch.distributed.all_reduce(query_amax, op=torch.distributed.ReduceOp.MAX, group=tp_group)
                torch.distributed.all_reduce(key_amax, op=torch.distributed.ReduceOp.MAX, group=tp_group)
                torch.distributed.all_reduce(value_amax, op=torch.distributed.ReduceOp.MAX, group=tp_group)

            if not self.training:
                self._qkv_query_amax_calibration.fill_(
                    max(self._qkv_query_amax_calibration.item(), query_amax.item())
                )
                self._qkv_key_amax_calibration.fill_(
                    max(self._qkv_key_amax_calibration.item(), key_amax.item())
                )
                self._qkv_value_amax_calibration.fill_(
                    max(self._qkv_value_amax_calibration.item(), value_amax.item())
                )

            query_scale = (query_amax / _FP8_E4M3_MAX).clamp_min(_TINY)
            key_scale = (key_amax / _FP8_E4M3_MAX).clamp_min(_TINY)
            value_scale = (value_amax / _FP8_E4M3_MAX).clamp_min(_TINY)
            if not use_qdq_flash_attn:
                query = self._fake_qkv_ste(query, scale=query_scale)
                key = self._fake_qkv_ste(key, scale=key_scale)
                value = self._fake_qkv_ste(value, scale=value_scale)

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        if use_qdq_flash_attn:
            if inference_context is not None and not inference_context.is_static_batching():
                raise ValueError(
                    "--attention-backend qdq_flash_attn does not support dynamic batching."
                )
            core_attn_out = self._qdq_flash_varlen_attention(
                query,
                key,
                value,
                query_scale,
                key_scale,
                value_scale,
                attn_mask_type,
                packed_seq_params,
            )
        elif self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # =================
        # Output. [sq, b, h]
        # =================
        nvtx_range_push(suffix="sigmoid_gate")
        core_attn_out = core_attn_out * torch.sigmoid(gate)
        nvtx_range_pop(suffix="sigmoid_gate")

        nvtx_range_push(suffix="linear_proj")
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix="linear_proj")

        return output, bias

    def get_qkv_calibration_scales(self) -> dict[str, float] | None:
        """Return calibrated QKV scales after eval, or None if not applicable."""
        if self.fake_qkv_quantization_type != "fp8_e4m3":
            return None
        query_amax = self._qkv_query_amax_calibration.item()
        key_amax = self._qkv_key_amax_calibration.item()
        value_amax = self._qkv_value_amax_calibration.item()
        if query_amax == 0.0 and key_amax == 0.0 and value_amax == 0.0:
            return None
        _FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
        return {
            "query_scale": query_amax / _FP8_E4M3_MAX,
            "key_scale": key_amax / _FP8_E4M3_MAX,
            "value_scale": value_amax / _FP8_E4M3_MAX,
        }


def collect_qkv_calibration_scales(model: torch.nn.Module) -> dict[str, float]:
    """Collect calibrated QKV FP8 scales from all GatedSoftmaxAttention layers.

    Call this after running eval to get per-layer scales for vLLM inference.
    Distinguishes decoder attention layers from MTP attention layers by checking
    the module hierarchy.

    Returns:
        Dict mapping "layer.{N}.{q,k,v}_scale" to float values,
        where N is the 0-indexed layer number. MTP layers use "mtp_layer.{N}" prefix.
    """
    from megatron_patch.model.qwen3_next.mamba_mtp import MultiTokenPredictionBlock

    scales = {}
    mtp_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, MultiTokenPredictionBlock):
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, GatedSoftmaxAttention):
                    mtp_modules.add(id(sub_module))

    mtp_idx = 0
    for module in model.modules():
        if isinstance(module, GatedSoftmaxAttention):
            layer_scales = module.get_qkv_calibration_scales()
            if layer_scales is not None:
                if id(module) in mtp_modules:
                    prefix = f"mtp_layer.{mtp_idx}"
                    mtp_idx += 1
                else:
                    layer_idx = module.layer_number - 1
                    prefix = f"layer.{layer_idx}"
                scales[f"{prefix}.q_scale"] = layer_scales["query_scale"]
                scales[f"{prefix}.k_scale"] = layer_scales["key_scale"]
                scales[f"{prefix}.v_scale"] = layer_scales["value_scale"]
    return scales
