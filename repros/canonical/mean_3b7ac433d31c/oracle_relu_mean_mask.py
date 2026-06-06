"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full SqueezeNet ReLU, 13x13 spatial mean, final contiguous `[512,1000]` view, and full `[512,1000,13,13]` boolean `relu <= 0` sibling output in one Triton reduction/store kernel while folding the boolean comparison to `x <= 0`, whereas Inductor currently schedules the small spatial reduction and the full-layout boolean side output as separate consumer kernels of the same ReLU producer; Inductor cannot do this today because its scheduler does not fuse a fixed-size reduction consumer and a full-layout pointwise side-output consumer into one multi-output loop nest with direct final-layout stores; the fix is SCHEDULER_FUSION: add a guarded producer-fanout fusion for small spatial reductions plus layout-preserving pointwise sibling stores."""
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


BATCH = 512
CHANNELS = 1000
HEIGHT = 13
WIDTH = 13
HW = HEIGHT * WIDTH
TOTAL_ROWS = BATCH * CHANNELS

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
MEAN_SHAPE = (BATCH, CHANNELS)
MEAN_STRIDE = (CHANNELS, 1)
MASK_SHAPE = INPUT_SHAPE
MASK_STRIDE = INPUT_STRIDE

BLOCK_ROWS = 8
BLOCK_HW = 256
INV_HW = 1.0 / HW


if triton is not None:

    @triton.jit
    def _relu_mean_mask_kernel(
        x_ptr,
        mean_ptr,
        mask_ptr,
        total_rows: tl.constexpr,
        hw_size: tl.constexpr,
        inv_hw: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw_size
        active = valid_rows[:, None] & valid_hw[None, :]

        offsets = row_offsets[:, None] * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)

        tl.store(mask_ptr + offsets, x <= 0.0, mask=active)

        relu = tl.where(x != x, x, tl.maximum(x, 0.0))
        relu = tl.where(valid_hw[None, :], relu, 0.0)
        reduced = tl.sum(relu, axis=1) * inv_hw
        tl.store(mean_ptr + row_offsets, reduced, mask=valid_rows)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, Any]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")
    convolution_25, shape_param = inputs
    x = _require_f32_tensor("convolution_25", convolution_25, INPUT_SHAPE, INPUT_STRIDE)
    _require_shape("_shape_param_0", shape_param, MEAN_SHAPE)
    return x, shape_param


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
        raise RuntimeError("Triton is required for oracle_relu_mean_mask.py")

    x, _shape_param = _validate_inputs(inputs)
    mean = torch.empty_strided(MEAN_SHAPE, MEAN_STRIDE, device=x.device, dtype=torch.float32)
    le_mask = torch.empty_strided(MASK_SHAPE, MASK_STRIDE, device=x.device, dtype=torch.bool)

    _relu_mean_mask_kernel[(triton.cdiv(TOTAL_ROWS, BLOCK_ROWS),)](
        x,
        mean,
        le_mask,
        total_rows=TOTAL_ROWS,
        hw_size=HW,
        inv_hw=INV_HW,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=1,
        num_stages=3,
    )
    return mean, le_mask


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
