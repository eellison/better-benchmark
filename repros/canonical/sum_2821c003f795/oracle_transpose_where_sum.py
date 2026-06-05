"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AlexNet/VGG full scope by streaming the shared bool/f32/bool [1024,4096] producer once, writing the returned transposed-view backing storage and accumulating the sibling [4096] column sum from the same `where(mask, full, mm * bool * 2)` values, whereas Inductor currently pays generic multi-output reduction/template overhead for the materialized layout side output plus reduction; Inductor cannot do this today because its scheduler does not form a single dense pointwise-side-output plus compatible reduction template for this graph shape, even though this row has only one reduction and two outputs; the fix is SCHEDULER_FUSION: add a full-scope dense side-output/reduction fusion that shares producer reads and emits the required strided view layout."""
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
ROWS = 1024
COLS = 4096
INPUT_SHAPE = (ROWS, COLS)
INPUT_STRIDE = (COLS, 1)
TRANSPOSE_SHAPE = (COLS, ROWS)
TRANSPOSE_STRIDE = (1, COLS)
SUM_SHAPE = (COLS,)
SUM_STRIDE = (1,)

if triton is not None:

    @triton.jit
    def _where_transpose_sum_kernel(
        arg19_ptr,
        mm_ptr,
        arg22_ptr,
        full_ptr,
        transpose_out_ptr,
        sum_out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        mask = (rows < M) & (cols < N)
        offsets = rows * N + cols

        gate = tl.load(arg19_ptr + offsets, mask=mask, other=0)
        mm_values = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        where_mask = tl.load(arg22_ptr + offsets, mask=mask, other=0)
        full_value = tl.load(full_ptr).to(tl.float32)

        scaled = mm_values.to(tl.float32) * tl.where(gate, 2.0, 0.0)
        values = tl.where(where_mask, full_value, scaled)
        values = tl.where(mask, values, 0.0)

        tl.store(transpose_out_ptr + offsets, values, mask=mask)
        partial = tl.sum(values, axis=0)
        sum_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.atomic_add(sum_out_ptr + sum_cols, partial, sem="relaxed", mask=sum_cols < N)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    arg19_1, mm_2, arg22_1, full, shape_param_0 = inputs
    arg19_1 = _expect_tensor("arg19_1", arg19_1, INPUT_SHAPE, INPUT_STRIDE, torch.bool)
    mm_2 = _expect_tensor("mm_2", mm_2, INPUT_SHAPE, INPUT_STRIDE, torch.float32)
    arg22_1 = _expect_tensor("arg22_1", arg22_1, INPUT_SHAPE, INPUT_STRIDE, torch.bool)
    full = _expect_tensor("full", full, (), (), torch.float32)
    if _shape_tuple(shape_param_0) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")

    return arg19_1, mm_2, arg22_1, full


def oracle_forward(inputs):
    """Run the full `where -> (permute, sum)` scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_transpose_where_sum.py")

    arg19_1, mm_2, arg22_1, full = _validate_inputs(inputs)
    transpose_out = torch.empty_strided(
        TRANSPOSE_SHAPE,
        TRANSPOSE_STRIDE,
        device=mm_2.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        SUM_SHAPE,
        SUM_STRIDE,
        device=mm_2.device,
        dtype=torch.float32,
    )
    sum_out.zero_()
    grid = (triton.cdiv(ROWS, 128), triton.cdiv(COLS, 64))
    _where_transpose_sum_kernel[grid](
        arg19_1,
        mm_2,
        arg22_1,
        full,
        transpose_out,
        sum_out,
        M=ROWS,
        N=COLS,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )
    return transpose_out, sum_out


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
