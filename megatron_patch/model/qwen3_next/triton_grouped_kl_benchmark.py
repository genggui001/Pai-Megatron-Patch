from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import torch

try:
    from .triton_grouped_kl import (
        HAS_TRITON,
        torch_reference_grouped_kl,
        triton_grouped_kl,
    )
except ImportError:
    from triton_grouped_kl import HAS_TRITON, torch_reference_grouped_kl, triton_grouped_kl


@dataclass
class BenchResult:
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
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


def _measure_latency_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> List[float]:
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))
        del out
    return latencies_ms


def _measure_peak_memory_mb(fn: Callable[[], torch.Tensor], device: torch.device) -> float:
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    del out
    return peak / (1024**2)


def _summarize(latencies_ms: Sequence[float], peak_memory_mb: float) -> BenchResult:
    ordered = sorted(latencies_ms)
    p50_idx = len(ordered) // 2
    p90_idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return BenchResult(
        latency_mean_ms=statistics.mean(ordered),
        latency_p50_ms=ordered[p50_idx],
        latency_p90_ms=ordered[p90_idx],
        peak_memory_mb=peak_memory_mb,
    )


def _benchmark_one(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> BenchResult:
    latencies_ms = _measure_latency_ms(fn, warmup=warmup, iters=iters)
    peak_memory_mb = _measure_peak_memory_mb(fn, device)
    return _summarize(latencies_ms, peak_memory_mb)


def _format_result(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    reverse: bool,
    ref_result: BenchResult,
    triton_result: BenchResult,
    max_abs_diff: float,
    mean_abs_diff: float,
    max_rel_diff: float,
) -> str:
    direction = "reverse" if reverse else "forward"
    memory_ratio = triton_result.peak_memory_mb / ref_result.peak_memory_mb
    latency_ratio = triton_result.latency_mean_ms / ref_result.latency_mean_ms
    return (
        f"shape={shape} dtype={dtype} direction={direction}\n"
        f"  precision: max_abs_diff={max_abs_diff:.6e} "
        f"mean_abs_diff={mean_abs_diff:.6e} max_rel_diff={max_rel_diff:.6e}\n"
        f"  torch:  mean={ref_result.latency_mean_ms:.3f}ms "
        f"p50={ref_result.latency_p50_ms:.3f}ms p90={ref_result.latency_p90_ms:.3f}ms "
        f"peak={ref_result.peak_memory_mb:.2f}MB\n"
        f"  triton: mean={triton_result.latency_mean_ms:.3f}ms "
        f"p50={triton_result.latency_p50_ms:.3f}ms p90={triton_result.latency_p90_ms:.3f}ms "
        f"peak={triton_result.peak_memory_mb:.2f}MB\n"
        f"  ratios: latency={latency_ratio:.3f}x memory={memory_ratio:.3f}x"
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton grouped KL against PyTorch.")
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=_parse_shape,
        default=[(256, 2, 4096), (1024, 2, 8192), (2048, 1, 18944)],
        help="Tensor shapes written as SxBxV.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=_parse_dtype,
        default=[torch.bfloat16],
        help="Input dtypes, e.g. bf16 fp16 fp32.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["forward", "reverse"],
        help="Which KL directions to benchmark: forward and/or reverse.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the benchmark.")
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in the current environment.")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    directions = _directions_from_args(args.directions)

    for shape in args.shapes:
        for dtype in args.dtypes:
            for reverse in directions:
                student = torch.randn(shape, device=device, dtype=dtype)
                teacher = torch.randn(shape, device=device, dtype=dtype)

                ref_output = torch_reference_grouped_kl(
                    student, teacher, temperature=args.temperature, reverse=reverse
                )
                triton_output = triton_grouped_kl(
                    student, teacher, temperature=args.temperature, reverse=reverse
                )
                diff = (ref_output - triton_output).abs()
                rel = diff / ref_output.abs().clamp_min(1e-6)

                ref_result = _benchmark_one(
                    lambda: torch_reference_grouped_kl(
                        student, teacher, temperature=args.temperature, reverse=reverse
                    ),
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                triton_result = _benchmark_one(
                    lambda: triton_grouped_kl(
                        student, teacher, temperature=args.temperature, reverse=reverse
                    ),
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                )

                print(
                    _format_result(
                        shape=shape,
                        dtype=dtype,
                        reverse=reverse,
                        ref_result=ref_result,
                        triton_result=triton_result,
                        max_abs_diff=diff.max().item(),
                        mean_abs_diff=diff.mean().item(),
                        max_rel_diff=rel.max().item(),
                    )
                )
                print("")


if __name__ == "__main__":
    main()
