"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet per-channel population var_mean, rsqrt side output, scalar-gain normalized activation, and keepdim mean side output in one shape-specialized Triton program per channel, whereas Inductor currently lowers the graph through its generic var_mean normalization schedule without retaining the 1536-element channel tile for the broadcast normalize and side-output stores; Inductor cannot do this today because the scheduler lacks a guarded reduction-consumer fusion template for fixed inner-size correction=0 channel normalization with multiple returned reduction consumers; the fix is SCHEDULER_FUSION: add a per-channel var_mean+rsqrt+scale-normalize template that fuses the dependent broadcast epilogue and emits the requested side-output layouts directly."""
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


CHANNELS = 3072
INNER_SIZE = 1536
EPS = 1.0e-5
GAIN_SCALE = 0.02551551815399144
OUT_SHAPE = (CHANNELS, INNER_SIZE, 1, 1)
OUT_STRIDE = (INNER_SIZE, 1, 1, 1)
GAIN_SHAPE = (CHANNELS, 1, 1, 1)
GAIN_STRIDE = (1, 1, 1, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
MEAN_SHAPE = (1, CHANNELS, 1)
MEAN_STRIDE = (CHANNELS, 1, 1)
BLOCK_K = 2048


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _channel_var_mean_norm_kernel(
        x_ptr,
        gain_ptr,
        invstd_ptr,
        out_ptr,
        mean_ptr,
        inner_size: tl.constexpr,
        eps: tl.constexpr,
        gain_scale: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K_)
        mask = offsets < inner_size
        base = channel * inner_size + offsets

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / inner_size
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / inner_size
        invstd = tl.rsqrt(var + eps)
        gain = tl.load(gain_ptr + channel).to(tl.float32) * gain_scale

        y = centered * invstd
        y = y * gain
        tl.store(invstd_ptr + channel, invstd)
        tl.store(mean_ptr + channel, mean)
        tl.store(out_ptr + base, y, mask=mask)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    x, gain, view_shape, out_shape = inputs
    x = _expect_tensor("arg229_1", x, OUT_SHAPE, OUT_STRIDE)
    gain = _expect_tensor("arg230_1", gain, GAIN_SHAPE, GAIN_STRIDE)
    if x.device != gain.device:
        raise ValueError(f"tensor inputs must be on the same device, got {x.device} and {gain.device}")
    if tuple(view_shape) != (1, CHANNELS, -1):
        raise ValueError(f"unexpected view shape parameter: {view_shape!r}")
    if tuple(out_shape) != OUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {out_shape!r}")
    return x, gain


@oracle_impl(hardware="H100", shapes="(T([3072, 1536, 1, 1], f32), T([3072, 1, 1, 1], f32), S([1, 3072, -1]), S([3072, 1536, 1, 1]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    x, gain = _validate_inputs(inputs)
    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=x.dtype)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    mean = torch.empty_strided(MEAN_SHAPE, MEAN_STRIDE, device=x.device, dtype=x.dtype)

    _channel_var_mean_norm_kernel[(CHANNELS,)](
        x,
        gain,
        invstd,
        out,
        mean,
        inner_size=INNER_SIZE,
        eps=EPS,
        gain_scale=GAIN_SCALE,
        BLOCK_K_=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    return invstd, out, mean


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
