# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
# Eval-only script: load checkpoint, run forward passes, export calibrated KV scales.

import json
import os
from functools import partial
from pathlib import Path

import datasets
datasets.disable_caching()

import torch
import torch._dynamo

from megatron.core.enums import ModelType
from model_provider import model_provider as base_model_provider

from megatron.training.arguments import core_transformer_config_from_args
from megatron_patch.arguments import get_patch_args
from megatron.training import initialize_megatron, get_args, get_timers, print_rank_0
from megatron.training.training import (
    setup_model_and_optimizer,
    evaluate,
)

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 32

from megatron_patch.model.qwen3_next.mamba_model import MambaModel
from megatron_patch.model.qwen3_next.layer_specs import get_qwen3_next_layer_spec, get_qwen3_mtp_block_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig
from megatron_patch.model.qwen3_next.gated_attention import collect_kv_calibration_scales

from qwen3_base_energon_dataloader_provider import train_valid_test_dataloaders_provider


def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building MAMBA model (eval-only) ...')
    if config is None:
        config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
        config.head_k_dim = args.mamba_state_dim
        config.head_v_dim = args.mamba_head_dim
        config.num_k_heads = args.mamba_num_groups
        config.num_v_heads = args.mamba_num_heads

        config.moe_expert_fake_type = args.moe_expert_fake_type
        config.moe_expert_fake_group_size = args.moe_expert_fake_group_size
        config.moe_expert_fake_activation_type = args.moe_expert_fake_activation_type
        config.moe_expert_fake_activation_group_size = args.moe_expert_fake_activation_group_size
        config.attention_fake_kv_quantization_type = args.attention_fake_kv_quantization_type

    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_qwen3_next_layer_spec(args, config),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        mtp_block_spec=get_qwen3_mtp_block_spec(args, config),
        pg_collection=pg_collection,
    )

    return model


model_provider = partial(base_model_provider, mamba_builder)


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    from megatron.core import parallel_state

    train_valid_test_dataloaders_provider.is_distributed = True

    initialize_megatron(
        extra_args_provider=get_patch_args,
        args_defaults={},
    )

    args = get_args()
    timers = get_timers()

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, ModelType.encoder_or_decoder
    )

    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    config = model[0].config if hasattr(model[0], 'config') else None

    print_rank_0("Loading validation data...")
    train_valid_test_dataloaders_provider.is_distributed = True
    train_data_iterator, valid_data_iterator, test_data_iterator = (
        train_valid_test_dataloaders_provider(
            [0, args.eval_iters * args.global_batch_size, 0]
        )
    )

    print_rank_0(f"Running evaluation for {args.eval_iters} iterations...")

    for model_module in model:
        model_module.eval()

    total_loss_dict, _, _ = evaluate(
        forward_step,
        valid_data_iterator,
        model,
        process_non_loss_data_func=None,
        config=config,
        verbose=True,
    )

    import math
    from megatron.training.utils import print_rank_last
    string = ' validation loss | '
    for key in total_loss_dict:
        val = total_loss_dict[key].item()
        ppl = math.exp(min(20, val))
        string += f'{key} value: {val:.6E} | {key} PPL: {ppl:.6E} | '
    print_rank_last(string)

    print_rank_0("Collecting KV calibration scales...")
    local_scales = {}
    for model_module in model:
        unwrapped = model_module
        while hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
        layer_scales = collect_kv_calibration_scales(unwrapped)
        local_scales.update(layer_scales)

    # Gather scales from all PP ranks to PP rank 0
    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    if pp_world_size > 1:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        # Serialize local_scales to gather across PP ranks
        import pickle
        local_bytes = pickle.dumps(local_scales)
        local_tensor = torch.tensor(
            list(local_bytes), dtype=torch.uint8, device='cuda'
        )
        # Gather sizes first
        size_tensor = torch.tensor([len(local_bytes)], dtype=torch.long, device='cuda')
        size_list = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(pp_world_size)]
        torch.distributed.all_gather(size_list, size_tensor, group=pp_group)

        max_size = max(s.item() for s in size_list)
        padded = torch.zeros(max_size, dtype=torch.uint8, device='cuda')
        padded[:len(local_bytes)] = local_tensor
        gathered = [torch.zeros(max_size, dtype=torch.uint8, device='cuda') for _ in range(pp_world_size)]
        torch.distributed.all_gather(gathered, padded, group=pp_group)

        all_scales = {}
        for i in range(pp_world_size):
            sz = size_list[i].item()
            remote_bytes = bytes(gathered[i][:sz].cpu().tolist())
            remote_scales = pickle.loads(remote_bytes)
            all_scales.update(remote_scales)
    else:
        all_scales = local_scales

    # Only one rank writes the output
    if parallel_state.get_data_parallel_rank() == 0 and \
       parallel_state.get_tensor_model_parallel_rank() == 0 and \
       pp_rank == 0:
        output_dir = args.save if args.save else "."
        output_path = os.path.join(output_dir, "kv_cache_scales.json")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_scales, f, indent=2)
        print(f"KV cache scales saved to: {output_path}")
        print(f"Total layers with scales: {len(all_scales) // 2}")
        for k, v in sorted(all_scales.items()):
            print(f"  {k}: {v:.8e}")

    print_rank_0("Evaluation complete.")
