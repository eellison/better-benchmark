"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete channels-last ConvNeXtV2 exact-GELU plus Global Response Normalization forward scope as the canonical three-stage norm template: reduce exact-GELU squared over each spatial tile into compact per-(N,C) sum-of-squares, reduce those L2 norms across channels per sample, then recompute exact-GELU in the affine residual GRN output store with the original output layout; the measured full-scope floor is at or near Inductor's current torch.compile decomposition rather than exposing a confirmed scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination, or new-pattern gap; the fix is BANDWIDTH_BOUND: record this GRN norm-template case as at-floor unless broader pointwise/reduction codegen or launch-overhead work improves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _choose_block_c(spatial: int, channels: int) -> int:
    if spatial <= 64:
        return min(64, _next_power_of_2(channels))
    if spatial <= 256:
        return min(64, _next_power_of_2(channels))
    if spatial <= 1024:
        return min(4, _next_power_of_2(channels))
    return 1


if triton is not None:

    @triton.jit
    def _gelu_spatial_norm_kernel(
        x_ptr,
        sumsq_ptr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channel_block = tl.program_id(1) * BLOCK_C
        s_offsets = tl.arange(0, BLOCK_S)
        c_offsets = channel_block + tl.arange(0, BLOCK_C)
        mask = (s_offsets[:, None] < spatial) & (c_offsets[None, :] < channels)
        offsets = batch * channels * spatial + s_offsets[:, None] * channels + c_offsets[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

        sum_sq = tl.sum(gelu * gelu, axis=0)
        tl.store(sumsq_ptr + batch * channels + c_offsets, sum_sq, mask=c_offsets < channels)

    @triton.jit
    def _channel_mean_inv_kernel(
        sumsq_ptr,
        inv_mean_ptr,
        channels: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C)
        mask = c_offsets < channels
        sumsq = tl.load(sumsq_ptr + batch * channels + c_offsets, mask=mask, other=0.0).to(tl.float32)
        mean_norm = tl.sum(tl.where(mask, tl.sqrt(sumsq), 0.0), axis=0) / channels
        tl.store(inv_mean_ptr + batch, 1.0 / (mean_norm + 1.0e-6))

    @triton.jit
    def _grn_output_kernel(
        bias_ptr,
        weight_ptr,
        x_ptr,
        sumsq_ptr,
        inv_mean_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        mask = offsets < total
        channel = offsets % channels
        batch = offsets // (channels * spatial)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
        sumsq = tl.load(sumsq_ptr + batch * channels + channel, mask=mask, other=0.0).to(tl.float32)
        norm = tl.sqrt(sumsq)
        inv_mean = tl.load(inv_mean_ptr + batch, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        out = gelu + bias + weight * gelu * norm * inv_mean
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    bias, weight, x = inputs
    if not isinstance(bias, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(bias)!r}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(weight)!r}")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"input 2 must be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"input 2 must be 4D NCHW, got shape={tuple(x.shape)}")

    batch, channels, height, width = x.shape
    expected_x_stride = (channels * height * width, 1, width * channels, channels)
    if tuple(x.stride()) != expected_x_stride:
        raise ValueError(f"expected channels-last NCHW stride {expected_x_stride}, got {tuple(x.stride())}")
    if tuple(bias.shape) != (channels,) or tuple(weight.shape) != (channels,):
        raise ValueError(
            f"bias/weight must be channel vectors of length {channels}, "
            f"got {tuple(bias.shape)} and {tuple(weight.shape)}"
        )
    if bias.dtype != torch.float32 or weight.dtype != torch.float32 or x.dtype != torch.float32:
        raise TypeError(f"expected float32 inputs, got {bias.dtype}, {weight.dtype}, {x.dtype}")
    if not (bias.is_cuda and weight.is_cuda and x.is_cuda):
        raise RuntimeError("this Triton oracle expects CUDA tensor inputs")
    if bias.device != x.device or weight.device != x.device:
        raise ValueError(f"all tensor inputs must share a device, got {bias.device}, {weight.device}, {x.device}")
    if not bias.is_contiguous() or not weight.is_contiguous():
        raise ValueError(f"bias and weight must be contiguous, got strides {bias.stride()} and {weight.stride()}")

    return bias, weight, x


@oracle_impl(hardware="H100", shapes="(T([2560], f32), T([2560], f32), T([128, 2560, 7, 7], f32, stride=(125440, 1, 17920, 2560)))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    bias, weight, x = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    spatial = height * width
    total = x.numel()
    block_c = _choose_block_c(spatial, channels)
    sumsq = torch.empty((batch, channels), device=x.device, dtype=torch.float32)
    inv_mean = torch.empty((batch,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=x.dtype)

    block_s = triton.next_power_of_2(spatial)
    _gelu_spatial_norm_kernel[(batch, triton.cdiv(channels, block_c))](
        x,
        sumsq,
        channels=channels,
        spatial=spatial,
        BLOCK_S=block_s,
        BLOCK_C=block_c,
        num_warps=8 if block_s >= 1024 else 4,
        num_stages=3,
    )

    _channel_mean_inv_kernel[(batch,)](
        sumsq,
        inv_mean,
        channels=channels,
        BLOCK_C=triton.next_power_of_2(channels),
        num_warps=8,
    )

    _grn_output_kernel[(triton.cdiv(total, 1024),)](
        bias,
        weight,
        x,
        sumsq,
        inv_mean,
        out,
        total=total,
        channels=channels,
        spatial=spatial,
        BLOCK_ELEMS=1024,
        num_warps=4,
        num_stages=3,
    )
    return out


# --- CLI entry point ---
def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
