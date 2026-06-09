"""Gap diagnosis (classification: BANDWIDTH_BOUND): this diagnostic oracle computes the complete BERT softmax-backward fragment returned by `Repro.forward`, including the f32 row sum over `arg337_1`, natural f32 `exp(arg287_1)`, explicit multiply/subtract epilogue, the returned `[20005, 16384]` transpose view, and the sibling `[20005]` column sum, while fusing output materialization with partial column reductions so the materialized transpose backing storage is not reread for the second reduction. The hand-written Triton schedule is still slower than the compiled path under `bench_oracle()`, so the attempted multi-output fusion is not a true floor for this repro. Inductor already does better for the required full-scope dense exp/materialize/reduce workload; the fix is BANDWIDTH_BOUND: do not queue a local scheduler change from this oracle unless a future exact-numerics full-scope implementation beats the compiled measurement."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None
    libdevice = None

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


BATCH0 = 128
BATCH1 = 128
ROWS = BATCH0 * BATCH1
VOCAB = 20005
BASE_SHAPE = (ROWS, VOCAB)
BASE_STRIDE = (VOCAB, 1)
PERMUTE_SHAPE = (VOCAB, ROWS)
PERMUTE_STRIDE = (1, VOCAB)
SUM_SHAPE = (VOCAB,)
SUM_STRIDE = (1,)

ROW_BLOCK_N = triton.next_power_of_2(VOCAB) if triton is not None else 32768
BLOCK_M = 64
BLOCK_N = 128
FINAL_BLOCK_C = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None and libdevice is not None:

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _row_sum_kernel(
        arg337_ptr,
        row_sum_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N_)
        mask = (row < ROWS_N) & (cols < VOCAB_N)
        values = tl.load(arg337_ptr + row * VOCAB_N + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(row_sum_ptr + row, tl.sum(values, axis=0), mask=row < ROWS_N)

    @triton.jit
    def _materialize_and_partial_sum_kernel(
        arg337_ptr,
        arg287_ptr,
        row_sum_ptr,
        out_base_ptr,
        partial_ptr,
        ROWS_N: tl.constexpr,
        VOCAB_N: tl.constexpr,
        NUM_M_TILES: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)

        rows = row_tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        cols = col_tile * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        row_mask = rows < ROWS_N
        col_mask = cols < VOCAB_N
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * VOCAB_N + cols[None, :]

        grad = tl.load(arg337_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        logits = tl.load(arg287_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_sums = tl.load(row_sum_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        exp_values = libdevice.exp(logits)
        scaled = _mul_rn_f32(exp_values, row_sums[:, None])
        values = _sub_rn_f32(grad, scaled)
        values = tl.where(mask, values, 0.0)

        tl.store(out_base_ptr + offsets, values, mask=mask)

        partial = tl.sum(values, axis=0)
        tl.store(
            partial_ptr + cols * NUM_M_TILES + row_tile,
            partial,
            mask=col_mask,
        )

    @triton.jit
    def _finalize_column_sum_kernel(
        partial_ptr,
        out_sum_ptr,
        VOCAB_N: tl.constexpr,
        NUM_M_TILES: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tiles = tl.arange(0, BLOCK_TILES_)
        mask = (cols[:, None] < VOCAB_N) & (tiles[None, :] < NUM_M_TILES)
        values = tl.load(
            partial_ptr + cols[:, None] * NUM_M_TILES + tiles[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(values, axis=1)
        tl.store(out_sum_ptr + cols, sums, mask=cols < VOCAB_N)


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


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    arg337_1, arg287_1, shape0, shape1 = inputs
    if _shape_tuple(shape0) != BASE_SHAPE:
        raise ValueError(f"unexpected first shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != SUM_SHAPE:
        raise ValueError(f"unexpected second shape parameter: {shape1!r}")

    arg337_1 = _expect_tensor("arg337_1", arg337_1, (BATCH0, BATCH1, VOCAB), (BATCH1 * VOCAB, VOCAB, 1), torch.float32)
    arg287_1 = _expect_tensor("arg287_1", arg287_1, (BATCH0, BATCH1, VOCAB), (BATCH1 * VOCAB, VOCAB, 1), torch.float32)
    return arg337_1, arg287_1


@oracle_impl(hardware="H100", shapes="(T([128, 128, 20005], f32), T([128, 128, 20005], f32), S([16384, 20005]), S([20005]))")
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
    arg337_1, arg287_1 = _validate_inputs(inputs)

    row_sums = torch.empty((ROWS,), device=arg337_1.device, dtype=torch.float32)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=arg337_1.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty_strided(
        SUM_SHAPE,
        SUM_STRIDE,
        device=arg337_1.device,
        dtype=torch.float32,
    )

    num_m_tiles = triton.cdiv(ROWS, BLOCK_M)
    partial = torch.empty((VOCAB, num_m_tiles), device=arg337_1.device, dtype=torch.float32)

    _row_sum_kernel[(ROWS,)](
        arg337_1,
        row_sums,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        BLOCK_N_=ROW_BLOCK_N,
        num_warps=8,
        num_stages=4,
    )
    _materialize_and_partial_sum_kernel[(num_m_tiles, triton.cdiv(VOCAB, BLOCK_N))](
        arg337_1,
        arg287_1,
        row_sums,
        out_base,
        partial,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        NUM_M_TILES=num_m_tiles,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )
    _finalize_column_sum_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_C),)](
        partial,
        out_sum,
        VOCAB_N=VOCAB,
        NUM_M_TILES=num_m_tiles,
        BLOCK_C_=FINAL_BLOCK_C,
        BLOCK_TILES_=triton.next_power_of_2(num_m_tiles),
        num_warps=4,
        num_stages=1,
    )

    return (out_base.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE), out_sum)


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
