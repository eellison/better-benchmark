"""
Oracle for var_mean_mean_34a86d931d28

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Repro.forward scope in one shape-specialized Triton groupnorm kernel, reducing each `[N, group]` row over 16 channels, applying affine scale/bias, adding the residual, applying ReLU, and writing the final contiguous `[64, 512]` result directly because the trailing spatial mean covers singleton `H=W=1` dimensions, whereas Inductor already lowers this tiny groupnorm-residual-mean graph to one fused norm-template kernel for the default shape; Inductor cannot materially improve it today through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute fusion because the measured cost is one launch plus the required f32 input/residual/affine reads and output write for a sub-megabyte workload; the fix class is BANDWIDTH_BOUND: record this as a diagnosis artifact unless this full-scope oracle beats the required tuned compile configs on the same shape.
"""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


NUM_GROUPS = 32
EPS = 1.0e-5


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
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=3),
        ],
        key=["total_groups", "channels", "hw_size", "group_elems"],
    )
    @triton.jit
    def _groupnorm_residual_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
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
        centered = x - mean[:, None]
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / count
        normalized = centered * tl.rsqrt(var[:, None] + eps)

        channels_per_group: tl.constexpr = channels // num_groups
        group_id = rows % num_groups
        channel = group_id[:, None] * channels_per_group + elems[None, :] // hw_size
        scale = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        y = tl.maximum(normalized * scale + shift + residual, 0.0)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    x, weight, bias, residual, group_shape, nchw_shape, out_shape = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError("expected tensor activation and residual inputs")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("expected tensor affine inputs")
    if x.ndim != 4 or residual.ndim != 4:
        raise ValueError(f"expected 4D activation/residual tensors, got {tuple(x.shape)} and {tuple(residual.shape)}")
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError(f"expected f32 activation/affine tensors, got {x.dtype}, {weight.dtype}, {bias.dtype}")
    if residual.dtype != torch.float32:
        raise TypeError(f"expected f32 residual tensor, got {residual.dtype}")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda or not residual.is_cuda:
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous() or not residual.is_contiguous():
        raise ValueError(
            "expected contiguous tensors, got "
            f"x_stride={x.stride()} weight_stride={weight.stride()} "
            f"bias_stride={bias.stride()} residual_stride={residual.stride()}"
        )

    n_batches, channels, height, width = (int(dim) for dim in x.shape)
    if tuple(residual.shape) != tuple(x.shape):
        raise ValueError(f"residual shape {tuple(residual.shape)} != activation shape {tuple(x.shape)}")
    if channels % NUM_GROUPS != 0:
        raise ValueError(f"channels must be divisible by {NUM_GROUPS}, got {channels}")
    if tuple(weight.shape) != (channels,) or tuple(bias.shape) != (channels,):
        raise ValueError(
            f"expected affine shape ({channels},), got weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )

    hw_size = height * width
    expected_group_shape = (n_batches, NUM_GROUPS, channels // NUM_GROUPS, hw_size)
    expected_nchw_shape = tuple(x.shape)
    expected_out_shape = (n_batches, channels)
    if not isinstance(group_shape, (list, tuple)) or tuple(group_shape) != expected_group_shape:
        raise ValueError(f"unexpected group view shape: {group_shape!r}, expected {expected_group_shape}")
    if not isinstance(nchw_shape, (list, tuple)) or tuple(nchw_shape) != expected_nchw_shape:
        raise ValueError(f"unexpected NCHW view shape: {nchw_shape!r}, expected {expected_nchw_shape}")
    if not isinstance(out_shape, (list, tuple)) or tuple(out_shape) != expected_out_shape:
        raise ValueError(f"unexpected output view shape: {out_shape!r}, expected {expected_out_shape}")
    if hw_size != 1:
        raise ValueError(f"{REPRO_ID} oracle is specialized to singleton spatial dims, got H*W={hw_size}")

    return x, weight, bias, residual, (n_batches, channels, height, width)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous [64, 512] float32 output tensor. The final
    mean([-1, -2], keepdim=True) is folded because H=W=1.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, weight, bias, residual, shape = _validate_inputs(inputs)
    n_batches, channels, height, width = shape
    hw_size = height * width
    group_elems = channels // NUM_GROUPS * hw_size
    total_groups = n_batches * NUM_GROUPS
    block_k = triton.next_power_of_2(group_elems)

    out_shape = (n_batches, channels)
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda meta: (triton.cdiv(total_groups, meta["BLOCK_M"]),)
    _groupnorm_residual_relu_kernel[grid](
        x,
        weight,
        bias,
        residual,
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


# --- CLI entry point ---
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
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
