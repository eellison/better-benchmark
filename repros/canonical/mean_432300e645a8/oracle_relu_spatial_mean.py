"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete SqueezeNet ReLU plus fixed 13x13 spatial mean and final contiguous [512,1000] view in one row-tiled Triton reduction kernel, while the required bench_oracle measurement shows tuned Inductor is already within the at-floor band for the same full scope; Inductor already emits a fused persistent relu/mean reduction for this norm-template-canonicalization case, so the remaining time is dominated by mandatory fp16 activation reads, ReLU, 169-element row reductions, and output stores rather than an avoidable local scheduler gap; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader reduction-template throughput work moves both oracle and compiled paths together."""
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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 512
CHANNELS = 1000
HEIGHT = 13
WIDTH = 13
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)


if triton is not None:

    from torch._inductor.runtime import triton_helpers

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 4}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=4, num_stages=3),
        ],
        key=["n_rows"],
    )
    @triton.jit
    def _relu_spatial_mean_kernel(
        x_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        spatial = tl.arange(0, BLOCK_HW)
        spatial_mask = spatial[None, :] < hw
        x = tl.load(
            x_ptr + rows[:, None] * hw + spatial[None, :],
            mask=spatial_mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        relu = triton_helpers.maximum(tl.full([1, 1], 0, tl.int32), x)
        relu = tl.where(spatial_mask, relu, 0.0)
        reduced = tl.sum(relu, axis=1) * (1.0 / 169.0)
        tl.store(out_ptr + rows, reduced)


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, view_shape = inputs
    x_t = _require_f16_tensor("convolution_25", x, INPUT_SHAPE, INPUT_STRIDE)
    _require_shape("_shape_param_0", view_shape, OUTPUT_SHAPE)
    return x_t


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full ReLU + spatial-mean + view computation returned by Repro."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_spatial_mean.py")

    x = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float16,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _relu_spatial_mean_kernel[grid](
        x,
        output,
        n_rows=ROWS,
        hw=HW,
        BLOCK_HW=256,
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
