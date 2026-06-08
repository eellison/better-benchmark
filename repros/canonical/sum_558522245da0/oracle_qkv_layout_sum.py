"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DeiT QKV layout clone backing storage for the returned `[2304,25344]` stride `(1,2304)` view and the sibling `[2304]` f32 column sum from the same q/k/v source tiles, whereas Inductor currently materializes the clone and rereads it through a separate 64-partial reduction; Inductor cannot do this today because its scheduler does not fuse a mandatory layout-changing clone producer with a compatible sibling reduction over that clone's rows; the fix is SCHEDULER_FUSION: teach layout materialization scheduling to emit same-tile reduction partials when the clone has a simple row-wise reduction consumer."""
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
HEADS = 12
TOKENS = 198
HEAD_DIM = 64
SOURCES = 3
ROWS = BATCH * TOKENS
SOURCE_COLS = HEADS * HEAD_DIM
COLS = SOURCES * SOURCE_COLS
ROWS_PER_PARTIAL = 396
NUM_PARTIALS = ROWS // ROWS_PER_PARTIAL

INPUT_SHAPE = (BATCH, HEADS, TOKENS, HEAD_DIM)
INPUT_STRIDE = (HEADS * TOKENS * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1)
SHAPE_0 = (SOURCES, BATCH, HEADS, TOKENS, HEAD_DIM)
SHAPE_1 = (BATCH, TOKENS, COLS)
SHAPE_2 = (ROWS, COLS)
SHAPE_3 = (COLS,)
TRANSPOSE_SHAPE = (COLS, ROWS)
TRANSPOSE_STRIDE = (1, COLS)

BLOCK_ROWS = 512
BLOCK_COLS = 16
FINAL_BLOCK_PARTIALS = 64
FINAL_BLOCK_COLS = 16


if triton is not None:

    @triton.jit
    def _layout_and_partial_sum_kernel(
        source_ptr,
        layout_ptr,
        partials_ptr,
        SOURCE: tl.constexpr,
        ROWS_PER_PARTIAL_: tl.constexpr,
        SOURCE_COLS_: tl.constexpr,
        COLS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        INPUT_S0: tl.constexpr,
        INPUT_S1: tl.constexpr,
        INPUT_S2: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
    ):
        partial_id = tl.program_id(0)
        col_block = tl.program_id(1)

        local_rows = tl.arange(0, BLOCK_ROWS_)
        local_cols = col_block * BLOCK_COLS_ + tl.arange(0, BLOCK_COLS_)
        rows = partial_id * ROWS_PER_PARTIAL_ + local_rows
        cols = SOURCE * SOURCE_COLS_ + local_cols

        row_mask = local_rows < ROWS_PER_PARTIAL_
        col_mask = local_cols < SOURCE_COLS_

        batch = rows // TOKENS_
        token = rows - batch * TOKENS_
        head = local_cols // HEAD_DIM_
        dim = local_cols - head * HEAD_DIM_

        input_offsets = (
            batch[:, None] * INPUT_S0
            + head[None, :] * INPUT_S1
            + token[:, None] * INPUT_S2
            + dim[None, :]
        )
        output_offsets = rows[:, None] * COLS_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        values = tl.load(source_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(layout_ptr + output_offsets, values, mask=mask)

        partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.store(
            partials_ptr + partial_id * COLS_ + cols,
            partial,
            mask=col_mask,
        )

    @triton.jit
    def _finish_sum_kernel(
        partials_ptr,
        sum_ptr,
        NUM_PARTIALS_: tl.constexpr,
        BLOCK_PARTIALS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_COLS_ + tl.arange(0, BLOCK_COLS_)
        partial_ids = tl.arange(0, BLOCK_PARTIALS_)
        mask = (partial_ids[:, None] < NUM_PARTIALS_) & (cols[None, :] < COLS_)
        values = tl.load(
            partials_ptr + partial_ids[:, None] * COLS_ + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(sum_ptr + cols, sums, mask=cols < COLS_)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected an iterable shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    q, k, v, shape0, shape1, shape2, shape3 = inputs
    tensors = (q, k, v)
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"input {index} must be a tensor, got {type(tensor).__name__}")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensors")
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {index} expected dtype torch.float32, got {tensor.dtype}")
        if tuple(tensor.shape) != INPUT_SHAPE:
            raise ValueError(f"input {index} expected shape {INPUT_SHAPE}, got {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != INPUT_STRIDE:
            raise ValueError(f"input {index} expected stride {INPUT_STRIDE}, got {tuple(tensor.stride())}")

    expected_shapes = (
        (shape0, SHAPE_0, "_shape_param_0"),
        (shape1, SHAPE_1, "_shape_param_1"),
        (shape2, SHAPE_2, "_shape_param_2"),
        (shape3, SHAPE_3, "_shape_param_3"),
    )
    for actual, expected, name in expected_shapes:
        actual_tuple = _shape_tuple(actual)
        if actual_tuple != expected:
            raise ValueError(f"{name} expected {expected}, got {actual_tuple}")

    return q, k, v


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

    q, k, v = _validate_inputs(inputs)

    layout = torch.empty_strided(SHAPE_2, (COLS, 1), device=q.device, dtype=torch.float32)
    partials = torch.empty((NUM_PARTIALS, COLS), device=q.device, dtype=torch.float32)
    sums = torch.empty_strided(SHAPE_3, (1,), device=q.device, dtype=torch.float32)

    grid = (NUM_PARTIALS, triton.cdiv(SOURCE_COLS, BLOCK_COLS))
    _layout_and_partial_sum_kernel[grid](
        q,
        layout,
        partials,
        SOURCE=0,
        ROWS_PER_PARTIAL_=ROWS_PER_PARTIAL,
        SOURCE_COLS_=SOURCE_COLS,
        COLS_=COLS,
        TOKENS_=TOKENS,
        HEAD_DIM_=HEAD_DIM,
        INPUT_S0=INPUT_STRIDE[0],
        INPUT_S1=INPUT_STRIDE[1],
        INPUT_S2=INPUT_STRIDE[2],
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_COLS_=BLOCK_COLS,
        num_warps=8,
        num_stages=4,
    )
    _layout_and_partial_sum_kernel[grid](
        k,
        layout,
        partials,
        SOURCE=1,
        ROWS_PER_PARTIAL_=ROWS_PER_PARTIAL,
        SOURCE_COLS_=SOURCE_COLS,
        COLS_=COLS,
        TOKENS_=TOKENS,
        HEAD_DIM_=HEAD_DIM,
        INPUT_S0=INPUT_STRIDE[0],
        INPUT_S1=INPUT_STRIDE[1],
        INPUT_S2=INPUT_STRIDE[2],
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_COLS_=BLOCK_COLS,
        num_warps=8,
        num_stages=4,
    )
    _layout_and_partial_sum_kernel[grid](
        v,
        layout,
        partials,
        SOURCE=2,
        ROWS_PER_PARTIAL_=ROWS_PER_PARTIAL,
        SOURCE_COLS_=SOURCE_COLS,
        COLS_=COLS,
        TOKENS_=TOKENS,
        HEAD_DIM_=HEAD_DIM,
        INPUT_S0=INPUT_STRIDE[0],
        INPUT_S1=INPUT_STRIDE[1],
        INPUT_S2=INPUT_STRIDE[2],
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_COLS_=BLOCK_COLS,
        num_warps=8,
        num_stages=4,
    )
    _finish_sum_kernel[(triton.cdiv(COLS, FINAL_BLOCK_COLS),)](
        partials,
        sums,
        NUM_PARTIALS_=NUM_PARTIALS,
        BLOCK_PARTIALS_=FINAL_BLOCK_PARTIALS,
        COLS_=COLS,
        BLOCK_COLS_=FINAL_BLOCK_COLS,
        num_warps=2,
        num_stages=1,
    )

    transposed = torch.as_strided(layout, TRANSPOSE_SHAPE, TRANSPOSE_STRIDE)
    return transposed, sums


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
