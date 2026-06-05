"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle streams the contiguous attention-head clone backing once, writing the returned [768,16384] transposed-view storage and accumulating the sibling [768] column sum from the same tiles, whereas Inductor currently emits a pointwise clone/permute kernel followed by separate partial and final reduction kernels over the cloned buffer; Inductor cannot do this today because its scheduler does not fuse a materialized layout-copy output with a sibling reduction consumer that needs the same producer elements; the fix is SCHEDULER_FUSION: add a multi-output layout-copy-plus-reduction schedule that writes the returned view backing and reduction partials from one producer traversal."""
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
BATCH = 32
HEADS = 12
SEQ = 512
HEAD_DIM = 64
ROWS = BATCH * SEQ
COLS = HEADS * HEAD_DIM

INPUT_SHAPE = (BATCH, HEADS, SEQ, HEAD_DIM)
INPUT_STRIDE = (SEQ * COLS, HEAD_DIM, COLS, 1)
VIEW0_SHAPE = (BATCH, SEQ, COLS)
VIEW1_SHAPE = (ROWS, COLS)
SUM_SHAPE = (COLS,)
TRANSPOSE_SHAPE = (COLS, ROWS)
TRANSPOSE_STRIDE = (1, COLS)

BLOCK_ROWS = 128
FINAL_BLOCK_COLS = 32


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_COLS": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 16}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 32}, num_warps=8, num_stages=4),
        ],
        key=["ROWS", "COLS"],
    )
    @triton.jit
    def _copy_partial_sum_kernel(
        x_ptr,
        transpose_out_ptr,
        partial_ptr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        col_tile = tl.program_id(0)
        row_tile = tl.program_id(1)

        rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
        offsets = rows[:, None] * COLS + cols[None, :]

        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(transpose_out_ptr + offsets, values, mask=mask)

        sums = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.store(partial_ptr + row_tile * COLS + cols, sums, mask=cols < COLS)


    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        sum_out_ptr,
        NUM_ROW_TILES: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        col_tile = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < COLS)

        values = tl.load(
            partial_ptr + tiles[:, None] * COLS + cols[None, :],
            mask=mask,
            other=0.0,
        )
        sums = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.store(sum_out_ptr + cols, sums, mask=cols < COLS)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected an iterable shape parameter, got {value!r}") from exc


def _require_input_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(value)!r}")
    if value.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA tensors")
    if value.dtype is not torch.float32:
        raise TypeError(f"input 0 must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != INPUT_SHAPE:
        raise ValueError(f"input 0 has shape {tuple(value.shape)}, expected {INPUT_SHAPE}")
    if tuple(value.stride()) != INPUT_STRIDE:
        raise ValueError(f"input 0 has stride {tuple(value.stride())}, expected {INPUT_STRIDE}")
    if value.storage_offset() != 0:
        raise ValueError(f"input 0 must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x = _require_input_tensor(inputs[0])
    if _shape_tuple(inputs[1]) != VIEW0_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {inputs[1]!r}")
    if _shape_tuple(inputs[2]) != VIEW1_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {inputs[2]!r}")
    if _shape_tuple(inputs[3]) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {inputs[3]!r}")
    return x


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
        raise RuntimeError("triton is required for this oracle")

    x = _validate_inputs(inputs)
    transpose_out = torch.empty_strided(
        TRANSPOSE_SHAPE,
        TRANSPOSE_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )
    sum_out = torch.empty_strided(SUM_SHAPE, (1,), device=x.device, dtype=x.dtype)

    num_row_tiles = triton.cdiv(ROWS, BLOCK_ROWS)
    partial = torch.empty((num_row_tiles, COLS), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_COLS"]), num_row_tiles)
    _copy_partial_sum_kernel[grid](
        x,
        transpose_out,
        partial,
        ROWS=ROWS,
        COLS=COLS,
        BLOCK_ROWS=BLOCK_ROWS,
    )
    _final_sum_kernel[(triton.cdiv(COLS, FINAL_BLOCK_COLS),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        COLS=COLS,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=4,
        num_stages=4,
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
