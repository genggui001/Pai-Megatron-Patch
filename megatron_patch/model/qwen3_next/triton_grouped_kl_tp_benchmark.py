from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

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


@dataclass
class BenchResult:
    forward_ms: float
    backward_ms: float
    total_ms: float
    peak_memory_mb: float


def _parse_shape(shape_text: str) -> Tuple[int, ...]:
    dims = tuple(int(part) for part in shape_text.lower().split("x"))
    if len(dims) < 2:
        raise argparse.ArgumentTypeError(f"invalid shape '{shape_text}', expected axbxc")
    return dims


def _parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise argparse.ArgumentTypeError(f"unsupported dtype '{dtype_name}'")
    return mapping[key]


def _directions_from_args(selected: Iterable[str]) -> List[bool]:
    result = []
    for item in selected:
        lowered = item.lower()
        if lowered == "forward":
            result.append(False)
        elif lowered == "reverse":
            result.append(True)
        else:
            raise ValueError(f"unsupported direction '{item}'")
    return result


def _init_dist() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _max_across_ranks(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _mean_across_ranks(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor.item()


def _free_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _benchmark_one(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    student_template: torch.Tensor,
    teacher: torch.Tensor,
    grad_weight: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> BenchResult:
    forward_latencies = []
    backward_latencies = []
    total_latencies = []

    for _ in range(warmup + iters):
        student = student_template.detach().clone().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_total = torch.cuda.Event(enable_timing=True)
        end_forward = torch.cuda.Event(enable_timing=True)
        end_backward = torch.cuda.Event(enable_timing=True)

        start_total.record()
        loss = loss_fn(student, teacher)
        end_forward.record()
        (loss.float() * grad_weight).sum().backward()
        end_backward.record()
        torch.cuda.synchronize()

        if _ >= warmup:
            total_ms = start_total.elapsed_time(end_backward)
            forward_ms = start_total.elapsed_time(end_forward)
            backward_ms = end_forward.elapsed_time(end_backward)
            total_latencies.append(total_ms)
            forward_latencies.append(forward_ms)
            backward_latencies.append(backward_ms)

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return BenchResult(
        forward_ms=statistics.mean(forward_latencies),
        backward_ms=statistics.mean(backward_latencies),
        total_ms=statistics.mean(total_latencies),
        peak_memory_mb=peak_memory_mb,
    )


def _format_result(
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    reverse: bool,
    world_size: int,
    output_diff: Tuple[float, float, float],
    grad_diff: Tuple[float, float, float],
    ref_result: BenchResult,
    triton_result: BenchResult,
) -> str:
    direction = "reverse" if reverse else "forward"
    return (
        f"tp={world_size} shape={shape} dtype={dtype} direction={direction}\n"
        f"  output_diff: max_abs={output_diff[0]:.6e} mean_abs={output_diff[1]:.6e} "
        f"max_rel={output_diff[2]:.6e}\n"
        f"  grad_diff:   max_abs={grad_diff[0]:.6e} mean_abs={grad_diff[1]:.6e} "
        f"max_rel={grad_diff[2]:.6e}\n"
        f"  torch:  forward={ref_result.forward_ms:.3f}ms backward={ref_result.backward_ms:.3f}ms "
        f"total={ref_result.total_ms:.3f}ms peak={ref_result.peak_memory_mb:.2f}MB\n"
        f"  triton: forward={triton_result.forward_ms:.3f}ms backward={triton_result.backward_ms:.3f}ms "
        f"total={triton_result.total_ms:.3f}ms peak={triton_result.peak_memory_mb:.2f}MB\n"
        f"  ratios: fwd={triton_result.forward_ms / ref_result.forward_ms:.3f}x "
        f"bwd={triton_result.backward_ms / ref_result.backward_ms:.3f}x "
        f"total={triton_result.total_ms / ref_result.total_ms:.3f}x "
        f"mem={triton_result.peak_memory_mb / ref_result.peak_memory_mb:.3f}x"
    )


def _format_triton_only_result(
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    reverse: bool,
    world_size: int,
    triton_result: BenchResult,
) -> str:
    direction = "reverse" if reverse else "forward"
    return (
        f"tp={world_size} shape={shape} dtype={dtype} direction={direction} [triton-only]\n"
        f"  triton: forward={triton_result.forward_ms:.3f}ms "
        f"backward={triton_result.backward_ms:.3f}ms "
        f"total={triton_result.total_ms:.3f}ms "
        f"peak={triton_result.peak_memory_mb:.2f}MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TP-aware Triton grouped KL.")
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=_parse_shape,
        default=[(256, 2, 4096), (1024, 1, 8192), (2048, 1, 18944)],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=_parse_dtype,
        default=[torch.bfloat16],
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["forward", "reverse"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--triton-only",
        action="store_true",
        help="Run only Triton benchmark without torch reference comparisons.",
    )
    args = parser.parse_args()

    if not HAS_TRITON:
        raise RuntimeError("Triton is required for TP benchmark.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TP benchmark.")

    rank, world_size, _ = _init_dist()
    directions = _directions_from_args(args.directions)
    device = torch.device("cuda")

    try:
        for shape in args.shapes:
            for dtype in args.dtypes:
                for reverse in directions:
                    torch.manual_seed(args.seed + rank)
                    student_template = torch.randn(shape, device=device, dtype=dtype)
                    teacher = torch.randn(shape, device=device, dtype=dtype)
                    torch.manual_seed(args.seed)
                    grad_weight = torch.randn(shape[:-1], device=device, dtype=torch.float32)

                    if args.triton_only:
                        tri_result = _benchmark_one(
                            lambda s, t: triton_grouped_kl_tp(
                                s,
                                t,
                                temperature=args.temperature,
                                reverse=reverse,
                                tp_group=dist.group.WORLD,
                            ),
                            student_template,
                            teacher,
                            grad_weight,
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        _free_cuda_memory()
                        tri_result = BenchResult(
                            forward_ms=_max_across_ranks(tri_result.forward_ms),
                            backward_ms=_max_across_ranks(tri_result.backward_ms),
                            total_ms=_max_across_ranks(tri_result.total_ms),
                            peak_memory_mb=_max_across_ranks(tri_result.peak_memory_mb),
                        )
                        if rank == 0:
                            print(
                                _format_triton_only_result(
                                    shape=shape,
                                    dtype=dtype,
                                    reverse=reverse,
                                    world_size=world_size,
                                    triton_result=tri_result,
                                )
                            )
                            print("")
                        continue

                    ref_student = student_template.detach().clone().requires_grad_(True)
                    ref_out = torch_reference_grouped_kl_tp(
                        ref_student,
                        teacher,
                        temperature=args.temperature,
                        reverse=reverse,
                        tp_group=dist.group.WORLD,
                    )
                    (ref_out.float() * grad_weight).sum().backward()
                    ref_grad = ref_student.grad.detach().clone()
                    ref_out_detached = ref_out.detach().clone()
                    del ref_student
                    del ref_out
                    _free_cuda_memory()

                    tri_student = student_template.detach().clone().requires_grad_(True)
                    tri_out = triton_grouped_kl_tp(
                        tri_student,
                        teacher,
                        temperature=args.temperature,
                        reverse=reverse,
                        tp_group=dist.group.WORLD,
                    )
                    (tri_out.float() * grad_weight).sum().backward()
                    tri_grad = tri_student.grad.detach().clone()
                    tri_out_detached = tri_out.detach().clone()
                    del tri_student
                    del tri_out
                    _free_cuda_memory()

                    out_diff = (ref_out_detached - tri_out_detached).abs()
                    out_mean = _mean_across_ranks(out_diff.mean().item())
                    out_max = _max_across_ranks(out_diff.max().item())
                    ref_out_norm = ref_out_detached.abs().max().clamp_min(1e-6).item()
                    out_rel = _max_across_ranks(
                        out_diff.div(ref_out_detached.abs().clamp_min(1e-6)).max().item()
                    )

                    grad_diff_tensor = (ref_grad - tri_grad).abs()
                    grad_mean = _mean_across_ranks(grad_diff_tensor.mean().item())
                    grad_max = _max_across_ranks(grad_diff_tensor.max().item())
                    ref_grad_norm = ref_grad.abs().max().clamp_min(1e-6).item()
                    grad_rel = _max_across_ranks(
                        grad_diff_tensor.div(ref_grad.abs().clamp_min(1e-6)).max().item()
                    )
                    del ref_grad
                    del tri_grad
                    del ref_out_detached
                    del tri_out_detached
                    _free_cuda_memory()

                    ref_result = _benchmark_one(
                        lambda s, t: torch_reference_grouped_kl_tp(
                            s,
                            t,
                            temperature=args.temperature,
                            reverse=reverse,
                            tp_group=dist.group.WORLD,
                        ),
                        student_template,
                        teacher,
                        grad_weight,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    _free_cuda_memory()
                    tri_result = _benchmark_one(
                        lambda s, t: triton_grouped_kl_tp(
                            s,
                            t,
                            temperature=args.temperature,
                            reverse=reverse,
                            tp_group=dist.group.WORLD,
                        ),
                        student_template,
                        teacher,
                        grad_weight,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    _free_cuda_memory()

                    ref_result = BenchResult(
                        forward_ms=_max_across_ranks(ref_result.forward_ms),
                        backward_ms=_max_across_ranks(ref_result.backward_ms),
                        total_ms=_max_across_ranks(ref_result.total_ms),
                        peak_memory_mb=_max_across_ranks(ref_result.peak_memory_mb),
                    )
                    tri_result = BenchResult(
                        forward_ms=_max_across_ranks(tri_result.forward_ms),
                        backward_ms=_max_across_ranks(tri_result.backward_ms),
                        total_ms=_max_across_ranks(tri_result.total_ms),
                        peak_memory_mb=_max_across_ranks(tri_result.peak_memory_mb),
                    )

                    if rank == 0:
                        print(
                            _format_result(
                                shape=shape,
                                dtype=dtype,
                                reverse=reverse,
                                world_size=world_size,
                                output_diff=(out_max, out_mean, out_rel),
                                grad_diff=(grad_max, grad_mean, grad_rel),
                                ref_result=ref_result,
                                triton_result=tri_result,
                            )
                        )
                        print("")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
