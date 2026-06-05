"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete guarded row reduction, divide/add epilogue, zero-filled `[64,50,256]` materialization, and `select_scatter(..., dim=1, index=-1)` store in one Triton kernel, whereas Inductor currently lowers the row reduction and the full/select_scatter materialization as separate generic scheduled work; Inductor cannot do this today because scheduler/codegen does not model a row-reduction producer feeding a structured zero-fill select_scatter side output as one fused scatter-reduce template; the fix is SCATTER_REDUCE: add a structured select_scatter-reduce lowering that keeps the row sum in registers while writing the sparse slice and required zero materialization directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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

ROWS = 64
DEPTH = 50
COLS = 256
SCATTER_INDEX = DEPTH - 1

if triton is not None:

    @triton.jit
    def _select_scatter_reduce_kernel(
        arg2_ptr,
        arg1_ptr,
        arg0_ptr,
        out_ptr,
        arg2_stride_0: tl.constexpr,
        arg2_stride_1: tl.constexpr,
        arg1_stride_0: tl.constexpr,
        arg1_stride_1: tl.constexpr,
        arg0_stride_0: tl.constexpr,
        arg0_stride_1: tl.constexpr,
        out_stride_0: tl.constexpr,
        out_stride_1: tl.constexpr,
        out_stride_2: tl.constexpr,
        DEPTH_SIZE: tl.constexpr,
        COLS_SIZE: tl.constexpr,
        SCATTER_INDEX_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < COLS_SIZE

        arg2 = tl.load(
            arg2_ptr + row * arg2_stride_0 + cols * arg2_stride_1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        arg0 = tl.load(
            arg0_ptr + row * arg0_stride_0 + cols * arg0_stride_1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        arg1 = tl.load(arg1_ptr + row * arg1_stride_0).to(tl.float32)

        denom = tl.maximum(arg1, 1.0e-12)
        row_terms = (-arg2) * ((arg0 / denom) / denom)
        row_sum = tl.sum(row_terms, axis=0)

        guarded_sum = tl.where(arg1 >= 1.0e-12, row_sum, 0.0)
        raw_div = arg0 / arg1
        guarded_div = tl.where(arg1 == 0.0, 0.0, raw_div)
        scatter_values = (arg2 / denom) + guarded_sum * guarded_div
        zero_values = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for depth in tl.static_range(0, DEPTH_SIZE):
            values = scatter_values if depth == SCATTER_INDEX_SIZE else zero_values
            tl.store(
                out_ptr
                + row * out_stride_0
                + depth * out_stride_1
                + cols * out_stride_2,
                values,
                mask=mask,
            )


def _validate_inputs(inputs):
    if len(inputs) < 3:
        raise ValueError("expected arg2, arg1, arg0, and shape parameter inputs")
    arg2, arg1, arg0 = inputs[:3]
    expected = ((ROWS, COLS), (ROWS, 1), (ROWS, COLS))
    actual = (tuple(arg2.shape), tuple(arg1.shape), tuple(arg0.shape))
    if actual != expected:
        raise ValueError(f"unexpected input shapes: {actual} != {expected}")
    return arg2, arg1, arg0


def _oracle_forward_torch(inputs):
    arg2, arg1, arg0 = _validate_inputs(inputs)
    full_zero = torch.full((), 0.0, dtype=torch.float32, device=arg0.device)
    denom = torch.clamp_min(arg1, 1e-12)
    row_terms = torch.mul(torch.neg(arg2), torch.div(torch.div(arg0, denom), denom))
    row_sum = torch.sum(row_terms, dim=1, keepdim=True)
    guarded_sum = torch.where(arg1 >= 1e-12, row_sum, full_zero)
    guarded_div = torch.where(arg1 == 0, full_zero, torch.div(arg0, arg1))
    scatter_values = torch.div(arg2, denom) + guarded_sum * guarded_div
    out = torch.empty_strided(
        (ROWS, DEPTH, COLS),
        (DEPTH * COLS, COLS, 1),
        dtype=torch.float32,
        device=arg0.device,
    )
    out.zero_()
    out.select(1, -1).copy_(scatter_values)
    return out


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
    arg2, arg1, arg0 = _validate_inputs(inputs)

    if triton is None or not arg0.is_cuda:
        return _oracle_forward_torch(inputs)

    output = torch.empty_strided(
        (ROWS, DEPTH, COLS),
        (DEPTH * COLS, COLS, 1),
        dtype=torch.float32,
        device=arg0.device,
    )
    _select_scatter_reduce_kernel[(ROWS,)](
        arg2,
        arg1,
        arg0,
        output,
        arg2.stride(0),
        arg2.stride(1),
        arg1.stride(0),
        arg1.stride(1),
        arg0.stride(0),
        arg0.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        DEPTH_SIZE=DEPTH,
        COLS_SIZE=COLS,
        SCATTER_INDEX_SIZE=SCATTER_INDEX,
        BLOCK_N=COLS,
        num_warps=8,
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
