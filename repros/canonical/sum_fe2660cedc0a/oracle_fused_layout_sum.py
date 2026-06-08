"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete masked f32 pointwise producer once, writes the returned transposed-view backing storage, and accumulates the sibling f32[768] column sum through a row-major 64-row partial reduction, whereas Inductor first materializes the producer and then launches two separate reduction kernels that reread it; Inductor cannot do this today because its scheduler does not fuse a required layout-changing side output with a compatible sibling reduction over the shared producer; the fix is SCHEDULER_FUSION: teach scheduler/codegen to emit a multi-output pointwise-plus-partial-reduction kernel when a materialized view output and a reduction consume the same expression."""
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


BATCH = 32
SEQ = 512
COLS = 768
ROWS = BATCH * SEQ
PAIR = 2
REDUCE_CHUNKS = 256
ROWS_PER_CHUNK = ROWS // REDUCE_CHUNKS

VIEW_AS_REAL_SHAPE = (BATCH, SEQ, COLS, PAIR)
VIEW_AS_REAL_STRIDE = (SEQ * COLS * PAIR, COLS * PAIR, PAIR, 1)
MUL_SHAPE = (BATCH, SEQ, COLS)
MUL_STRIDE = (SEQ * COLS, COLS, 1)
MASK_SHAPE = MUL_SHAPE
MASK_STRIDE = MUL_STRIDE
BASE_SHAPE = (ROWS, COLS)
BASE_STRIDE = (COLS, 1)
OUT0_SHAPE = (COLS, ROWS)
OUT0_STRIDE = (1, COLS)
PARTIAL_SHAPE = (1, COLS, REDUCE_CHUNKS)
PARTIAL_STRIDE = (COLS * REDUCE_CHUNKS, 1, COLS)
SUM_2D_SHAPE = (1, COLS)
SUM_2D_STRIDE = (COLS, 1)
OUT1_SHAPE = (COLS,)
OUT1_STRIDE = (1,)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=3),
        ],
        key=["COLS_N"],
    )
    @triton.jit
    def _pointwise_partial_sum_kernel(
        view_as_real_ptr,
        mul_ptr,
        mask_ptr,
        base_out_ptr,
        partial_ptr,
        COLS_N: tl.constexpr,
        ROWS_PER_CHUNK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        chunk = tl.program_id(1)

        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = chunk * ROWS_PER_CHUNK_N + tl.arange(0, ROWS_PER_CHUNK_N)
        offsets = rows[:, None] * COLS_N + cols[None, :]
        mask = cols[None, :] < COLS_N

        tmp0 = tl.load(mul_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tmp1 = tl.load(
            view_as_real_ptr + (offsets * 2),
            mask=mask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp3 = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full([1], 1.0, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp2 * tmp6

        tl.store(base_out_ptr + offsets, tmp7, mask=mask)
        partial = tl.sum(tmp7, axis=0)
        tl.store(partial_ptr + cols + chunk * COLS_N, partial, mask=cols < COLS_N)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 64}, num_warps=8, num_stages=3),
        ],
        key=["COLS_N"],
    )
    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        COLS_N: tl.constexpr,
        CHUNKS_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        col_mask = cols < COLS_N
        chunks = tl.arange(0, CHUNKS_N)[None, :]
        values = tl.load(
            partial_ptr + cols + chunks * COLS_N,
            mask=col_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        acc = tl.sum(values, axis=1)[:, None]
        tl.store(out_ptr + cols, acc, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_fused_layout_sum.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    view_as_real_11, mul_332, arg59_1, shape0, shape1 = inputs
    if _shape_tuple(shape0) != BASE_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != OUT1_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {shape1!r}")

    view_as_real_11 = _expect_tensor(
        "view_as_real_11",
        view_as_real_11,
        VIEW_AS_REAL_SHAPE,
        VIEW_AS_REAL_STRIDE,
        torch.float32,
    )
    mul_332 = _expect_tensor("mul_332", mul_332, MUL_SHAPE, MUL_STRIDE, torch.float32)
    arg59_1 = _expect_tensor("arg59_1", arg59_1, MASK_SHAPE, MASK_STRIDE, torch.bool)
    if mul_332.device != view_as_real_11.device or arg59_1.device != view_as_real_11.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return view_as_real_11, mul_332, arg59_1


def oracle_forward(inputs):
    """Run the complete Repro.forward scope with fused producer materialization and partial sum."""
    view_as_real_11, mul_332, arg59_1 = _validate_inputs(inputs)

    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mul_332.device,
        dtype=torch.float32,
    )
    partial = torch.empty_strided(
        PARTIAL_SHAPE,
        PARTIAL_STRIDE,
        device=mul_332.device,
        dtype=torch.float32,
    )
    sum_2d = torch.empty_strided(
        SUM_2D_SHAPE,
        SUM_2D_STRIDE,
        device=mul_332.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_C"]), REDUCE_CHUNKS)
    _pointwise_partial_sum_kernel[grid](
        view_as_real_11,
        mul_332,
        arg59_1,
        out_base,
        partial,
        COLS_N=COLS,
        ROWS_PER_CHUNK_N=ROWS_PER_CHUNK,
    )
    _final_sum_kernel[(lambda meta: (triton.cdiv(COLS, meta["BLOCK_C"]),))](
        partial,
        sum_2d,
        COLS_N=COLS,
        CHUNKS_N=REDUCE_CHUNKS,
    )
    return out_base.permute(1, 0), sum_2d.view(OUT1_SHAPE)


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
