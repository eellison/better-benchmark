"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured group-normalization affine ReLU from Repro.forward in one shape-specialized Triton row kernel, reducing each `[N, 32]` group over the full per-group channel/spatial extent, keeping mean and inverse standard deviation in registers, applying the per-channel scale and bias, and writing the full contiguous f32 NCHW output, whereas Inductor already lowers this tiny groupnorm-ish graph to a single fused kernel for the default shape; Inductor cannot materially improve it today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because one kernel launch plus the required input/affine reads and output write dominate the 0.3 MB workload; the fix class is BANDWIDTH_BOUND: record this as a diagnosis artifact unless this full-scope oracle beats the required compile configs on the same shape."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

NUM_GROUPS = 32
EPS = 1.0e-5
HISTORICAL_BEST_COMPILE_US = 5.727999843657017


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=3),
        ],
        key=["total_groups", "channels", "hw_size", "group_elems"],
    )
    @triton.jit
    def _groupnorm_affine_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_groups: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        group_elems: tl.constexpr,
        num_groups: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        elems = tl.arange(0, BLOCK_K)
        mask = (rows[:, None] < total_groups) & (elems[None, :] < group_elems)

        offsets = rows[:, None] * group_elems + elems[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        count = group_elems + 0.0
        mean = tl.sum(x, axis=1) / count
        sq_mean = tl.sum(x * x, axis=1) / count
        var = tl.maximum(sq_mean - mean * mean, 0.0)
        normalized = (x - mean[:, None]) * tl.rsqrt(var[:, None] + eps)

        group_id = rows % num_groups
        channels_per_group: tl.constexpr = channels // num_groups
        channel = group_id[:, None] * channels_per_group + elems[None, :] // hw_size
        scale = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(normalized * scale + shift, 0.0)

        tl.store(out_ptr + offsets, y, mask=mask)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, weight, bias, group_shape, out_shape = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("expected tensor affine inputs")
    if x.ndim != 4:
        raise ValueError(f"expected 4D input, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"expected f32 tensors, got {x.dtype}, {weight.dtype}, {bias.dtype}")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("oracle_groupnorm_relu.py expects CUDA tensors")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError(
            "expected contiguous tensors, got "
            f"x_stride={x.stride()} weight_stride={weight.stride()} bias_stride={bias.stride()}"
        )

    n_batches, channels, height, width = (int(dim) for dim in x.shape)
    if channels % NUM_GROUPS != 0:
        raise ValueError(f"channels must be divisible by {NUM_GROUPS}, got {channels}")
    if tuple(weight.shape) != (channels,) or tuple(bias.shape) != (channels,):
        raise ValueError(
            f"expected affine shape ({channels},), got weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )

    expected_group_shape = (n_batches, NUM_GROUPS, channels // NUM_GROUPS, height * width)
    if not isinstance(group_shape, (list, tuple)) or tuple(group_shape) != expected_group_shape:
        raise ValueError(f"unexpected group view shape: {group_shape!r}, expected {expected_group_shape}")
    if not isinstance(out_shape, (list, tuple)) or tuple(out_shape) != tuple(x.shape):
        raise ValueError(f"unexpected output view shape: {out_shape!r}, expected {tuple(x.shape)}")

    return x, weight, bias, (n_batches, channels, height, width)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward groupnorm affine ReLU computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_groupnorm_relu.py")

    x, weight, bias, shape = _validate_inputs(inputs)
    n_batches, channels, height, width = shape
    hw_size = height * width
    group_elems = channels // NUM_GROUPS * hw_size
    total_groups = n_batches * NUM_GROUPS
    block_k = triton.next_power_of_2(group_elems)

    out = torch.empty_strided(
        tuple(x.shape),
        _contiguous_stride(tuple(x.shape)),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda meta: (triton.cdiv(total_groups, meta["BLOCK_M"]),)
    _groupnorm_affine_relu_kernel[grid](
        x,
        weight,
        bias,
        out,
        total_groups=total_groups,
        channels=channels,
        hw_size=hw_size,
        group_elems=group_elems,
        num_groups=NUM_GROUPS,
        eps=EPS,
        BLOCK_K=block_k,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()
    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and expected.stride() == actual.stride()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        ok = ok and _check_layout(instance, inputs)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.3f}")
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
