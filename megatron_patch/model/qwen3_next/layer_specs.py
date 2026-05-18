# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
    TEColumnParallelLinear,
    TENorm
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.multi_token_prediction import (
    get_mtp_layer_offset,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron_patch.model.qwen3_next.gated_attention import GatedSoftmaxAttention
from megatron_patch.model.qwen3_next.mamba_block import MambaStack, MambaStackSubmodules
from megatron_patch.model.qwen3_next.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron_patch.model.qwen3_next.mamba_mtp import MultiTokenPredictionLayer, MultiTokenPredictionLayerSubmodules
from megatron_patch.model.qwen3_next.mamba_mtp import MultiTokenPredictionBlockSubmodules
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron_patch.model.qwen3_next.gated_deltanet import GatedDeltaNetMixer


def _module_spec_with_params(spec_or_module, **extra_params):
    if isinstance(spec_or_module, ModuleSpec):
        return ModuleSpec(
            module=spec_or_module.module,
            params={**spec_or_module.params, **extra_params},
            submodules=spec_or_module.submodules,
            metainfo=spec_or_module.metainfo,
        )
    return ModuleSpec(module=spec_or_module, params=extra_params)


def _resolve_moe_expert_fake_quantization(
    moe_expert_fake_type: Optional[str],
    moe_expert_fake_group_size: int,
):
    if moe_expert_fake_type is None:
        if moe_expert_fake_group_size > 0:
            raise ValueError("moe_expert_fake_group_size requires moe_expert_fake_type to be set.")
        return None, 0

    if moe_expert_fake_group_size <= 0:
        raise ValueError(
            "moe_expert_fake_group_size must be > 0 when fake quantization is enabled."
        )

    return moe_expert_fake_type, moe_expert_fake_group_size


def _maybe_enable_fake_quantization_for_te_grouped_experts(
    expert_module,
    expert_submodule,
    moe_expert_fake_type: Optional[str],
    moe_expert_fake_group_size: int,
):
    fake_type, fake_group_size = _resolve_moe_expert_fake_quantization(
        moe_expert_fake_type,
        moe_expert_fake_group_size,
    )

    if fake_type is None:
        return expert_submodule
    if expert_module is not TEGroupedMLP or expert_submodule is None:
        return expert_submodule
    if fake_group_size <= 0:
        raise ValueError(
            "moe_expert_fake_group_size must be > 0 when fake quantization is enabled."
        )

    fake_quant_params = {
        "fake_type": fake_type,
        "fake_group_size": fake_group_size,
    }
    return MLPSubmodules(
        linear_fc1=_module_spec_with_params(expert_submodule.linear_fc1, **fake_quant_params),
        activation_func=expert_submodule.activation_func,
        linear_fc2=_module_spec_with_params(expert_submodule.linear_fc2, **fake_quant_params),
    )

def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    moe_expert_fake_type: Optional[str] = None,
    moe_expert_fake_group_size: int = 0,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        moe_expert_fake_type=moe_expert_fake_type,
        moe_expert_fake_group_size=moe_expert_fake_group_size,
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
    moe_expert_fake_type: Optional[str] = None,
    moe_expert_fake_group_size: int = 0,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )
    if expert_submodule is not None:
        expert_submodule.activation_func = activation_func
        expert_submodule = _maybe_enable_fake_quantization_for_te_grouped_experts(
            expert_module,
            expert_submodule,
            moe_expert_fake_type,
            moe_expert_fake_group_size,
        )

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(experts=experts, shared_experts=shared_experts),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )
    return moe_module_spec


def get_qwen3_mtp_block_spec(args, config):
    if args.mtp_num_layers is None:
        return None

    num_layers_to_build = get_mtp_num_layers_to_build(config, vp_stage=None, pp_rank=None)
    if num_layers_to_build == 0:
        return None

    transformer_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=GatedSoftmaxAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=get_moe_module_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=args.moe_grouped_gemm,
                moe_expert_fake_type=None,
                moe_expert_fake_group_size=0,
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )

    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=TENorm,
            hnorm=TENorm,
            eh_proj=TEColumnParallelLinear,
            transformer_layer=transformer_layer_spec,
            layer_norm=TENorm,
        ),
    )

    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset : offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        assert (
            len(mtp_layer_specs) == config.mtp_num_layers
        ), +f"currently all of the mtp layers must stage in the same pipeline stage."
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
    else:
        mtp_block_spec = None

    return mtp_block_spec


def get_qwen3_next_layer_spec(args, config):
    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=GatedDeltaNetMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py (with MLP removed)
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=GatedSoftmaxAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            q_layernorm=TENorm,
                            k_layernorm=TENorm
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            mlp_layer = ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm,
                    mlp=get_moe_module_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        moe_expert_fake_type=getattr(config, "moe_expert_fake_type", None),
                        moe_expert_fake_group_size=getattr(
                            config, "moe_expert_fake_group_size", 0
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add
                ),
            ),
        ),
    )

