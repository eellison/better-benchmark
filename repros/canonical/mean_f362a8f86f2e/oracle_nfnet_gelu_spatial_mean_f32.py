"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet exact-erf GELU scale and 6x6 spatial mean for f32 `[128,3072,6,6]` in one Triton reduction kernel that writes the final contiguous f32 `[128,3072]` view directly, whereas Inductor currently lowers the activation producer and spatial mean through a generic reduction schedule with higher indexing and scheduling overhead for this fixed small-spatial case; Inductor cannot do this today because its scheduler/codegen does not expose a guarded exact-erf GELU producer fused into a fixed 6x6 spatial-mean template; the fix is SCHEDULER_FUSION: add a benchmark-gated pointwise-producer-to-small-spatial-mean fusion template that emits the direct row reduction and final viewed store."""
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


# --- Oracle kernel(s) ---

BATCH = 128
CHANNELS = 3072
HEIGHT = 6
WIDTH = 6
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CONTIGUOUS_X_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
CHANNELS_LAST_X_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
RSQRT2 = 0.7071067811865476
GAMMA = 1.7015043497085571
INV_HW = 1.0 / HW
CLASSIFICATION = "SCHEDULER_FUSION"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 64}, num_warps=8, num_stages=3),
        ],
        key=["x_stride_n", "x_stride_c", "x_stride_h", "x_stride_w"],
    )
    @triton.jit
    def _gelu_spatial_mean_f32_kernel(
        x_ptr,
        out_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        rsqrt2: tl.constexpr,
        gamma: tl.constexpr,
        inv_hw: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        n_offsets = row_offsets // channels
        c_offsets = row_offsets - n_offsets * channels
        valid_rows = row_offsets < total_rows

        hw0_offsets = tl.arange(0, 32)
        h0_offsets = hw0_offsets // width
        w0_offsets = hw0_offsets - h0_offsets * width
        x0_offsets = (
            n_offsets[:, None] * x_stride_n
            + c_offsets[:, None] * x_stride_c
            + h0_offsets[None, :] * x_stride_h
            + w0_offsets[None, :] * x_stride_w
        )
        x0 = tl.load(x_ptr + x0_offsets, mask=valid_rows[:, None], other=0.0).to(tl.float32)
        erf0 = tl.math.erf(x0 * rsqrt2) + 1.0
        gelu0 = ((x0 * 0.5) * erf0) * gamma
        sum0 = tl.sum(gelu0, axis=1)

        hw1_offsets = tl.arange(0, 4) + 32
        h1_offsets = hw1_offsets // width
        w1_offsets = hw1_offsets - h1_offsets * width
        x1_offsets = (
            n_offsets[:, None] * x_stride_n
            + c_offsets[:, None] * x_stride_c
            + h1_offsets[None, :] * x_stride_h
            + w1_offsets[None, :] * x_stride_w
        )
        x1 = tl.load(x_ptr + x1_offsets, mask=valid_rows[:, None], other=0.0).to(tl.float32)
        erf1 = tl.math.erf(x1 * rsqrt2) + 1.0
        gelu1 = ((x1 * 0.5) * erf1) * gamma
        reduced = (sum0 + tl.sum(gelu1, axis=1)) * inv_hw

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_input_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(
            f"{REPRO_ID} input shape {tuple(value.shape)} != "
            f"{(BATCH, CHANNELS, HEIGHT, WIDTH)}"
        )
    if value.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA input")
    allowed_strides = (CONTIGUOUS_X_STRIDE, CHANNELS_LAST_X_STRIDE)
    if tuple(value.stride()) not in allowed_strides:
        raise ValueError(
            f"{REPRO_ID} input stride {tuple(value.stride())} not in {allowed_strides}"
        )
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, view_shape = inputs
    x_t = _require_input_tensor(x)
    output_shape = tuple(int(dim) for dim in view_shape)
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} got unexpected view shape parameter: {view_shape!r}")
    return x_t, output_shape


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
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
        raise RuntimeError("Triton is required for oracle_nfnet_gelu_spatial_mean_f32.py")

    x, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _gelu_spatial_mean_f32_kernel[grid](
        x,
        output,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        total_rows=ROWS,
        channels=CHANNELS,
        width=WIDTH,
        rsqrt2=RSQRT2,
        gamma=GAMMA,
        inv_hw=INV_HW,
    )
    return output


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
