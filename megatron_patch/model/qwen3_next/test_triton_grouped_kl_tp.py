from __future__ import annotations

import os

import torch
import torch.distributed as dist

try:
    from .triton_grouped_kl import (
        HAS_TRITON,
        torch_reference_grouped_kl_tp,
        triton_grouped_kl_tp,
    )
except ImportError:
    from triton_grouped_kl import (
        HAS_TRITON,
        torch_reference_grouped_kl_tp,
        triton_grouped_kl_tp,
    )
def _init_process_group() -> tuple[int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size


def _max_across_ranks(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def main() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for TP tests.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TP tests.")

    rank, world_size = _init_process_group()
    device = torch.device("cuda")
    atol_rtol_cases = [
        ((33, 2, 1024), torch.bfloat16, 2e-2, 2e-2),
        ((17, 1, 513), torch.float32, 1e-5, 1e-5),
    ]

    try:
        for reverse in (False, True):
            for shape, dtype, atol, rtol in atol_rtol_cases:
                torch.manual_seed(1000 + rank)
                reference_student = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                triton_student = reference_student.detach().clone().requires_grad_(True)
                teacher = torch.randn(shape, device=device, dtype=dtype)
                torch.manual_seed(4242)
                grad_weight = torch.randn(shape[:-1], device=device, dtype=torch.float32)

                reference_loss = torch_reference_grouped_kl_tp(
                    reference_student,
                    teacher,
                    temperature=0.85,
                    reverse=reverse,
                    tp_group=dist.group.WORLD,
                )
                triton_loss = triton_grouped_kl_tp(
                    triton_student,
                    teacher,
                    temperature=0.85,
                    reverse=reverse,
                    tp_group=dist.group.WORLD,
                )

                loss_diff = (reference_loss - triton_loss).abs()
                max_loss_diff = _max_across_ranks(loss_diff.max().item())
                ref_loss_norm = reference_loss.abs().max().clamp_min(1e-6).item()
                max_loss_rel = _max_across_ranks(max_loss_diff / ref_loss_norm)

                (reference_loss.float() * grad_weight).sum().backward()
                (triton_loss.float() * grad_weight).sum().backward()
                grad_diff = (reference_student.grad - triton_student.grad).abs()
                max_grad_diff = _max_across_ranks(grad_diff.max().item())
                ref_grad_norm = reference_student.grad.abs().max().clamp_min(1e-6).item()
                max_grad_rel = _max_across_ranks(max_grad_diff / ref_grad_norm)

                if rank == 0:
                    print(
                        f"reverse={reverse} shape={shape} dtype={dtype} world_size={world_size} "
                        f"max_loss_diff={max_loss_diff:.6e} max_loss_rel={max_loss_rel:.6e} "
                        f"max_grad_diff={max_grad_diff:.6e} max_grad_rel={max_grad_rel:.6e}"
                    )

                assert max_loss_diff <= atol + rtol * max(ref_loss_norm, 1.0)
                assert max_loss_rel <= max(rtol, atol)
                assert max_grad_diff <= atol + rtol * max(ref_grad_norm, 1.0)
                assert max_grad_rel <= max(rtol, atol)

        if rank == 0:
            print("TP grouped KL forward/backward tests passed.")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
