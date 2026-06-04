"""
Oracle for sum_sum_sum_766837223ded

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the
row-wise layernorm-backward algebra with the side-output materialization and
then computes the three column reductions in one Triton reduction kernel,
whereas Inductor emits a less direct multi-kernel schedule for the dependent
row reductions, side transpose, and independent column reductions; Inductor
cannot do this today because its scheduler does not form a custom full-scope
multi-output reduction plan that reuses the row-local scalars across the
materialized transpose and the later reductions; the fix is SCHEDULER_FUSION:
teach the scheduler/codegen to group this dependent row-reduction plus
transpose plus column-reduction pattern into a small coordinated kernel set.
"""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


N_ROWS = 128
N_COLS = 768


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _side_transpose_kernel(
        mm_ptr,
        weight_ptr,
        addmm_ptr,
        mean_ptr,
        rstd_ptr,
        select_ptr,
        out2_ptr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < 768

        x = tl.load(mm_ptr + row * 768 + offs, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        addmm = tl.load(addmm_ptr + row * 768 + offs, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        select = tl.load(select_ptr + row * 768 + offs, mask=mask, other=0.0)

        mul = x * weight
        yhat = (addmm - mean) * rstd
        row_sum = tl.sum(mul, axis=0)
        row_dot = tl.sum(mul * yhat, axis=0)
        grad = (rstd / 768.0) * (mul * 768.0 - row_sum - yhat * row_dot)
        side = select + grad

        tl.store(out2_ptr + offs * 128 + row, side, mask=mask)

    @triton.jit
    def _column_reduction_kernel(
        mm_ptr,
        addmm_ptr,
        mean_ptr,
        rstd_ptr,
        out2_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)[:, None]
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
        mask = cols < 768

        x = tl.load(mm_ptr + rows * 768 + cols, mask=mask, other=0.0)
        addmm = tl.load(addmm_ptr + rows * 768 + cols, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + rows)
        rstd = tl.load(rstd_ptr + rows)
        side = tl.load(out2_ptr + cols * 128 + rows, mask=mask, other=0.0)

        yhat = (addmm - mean) * rstd
        sum_xyhat = tl.sum(x * yhat, axis=0)
        sum_x = tl.sum(x, axis=0)
        sum_side = tl.sum(side, axis=0)

        out_cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        out_mask = out_cols < 768
        tl.store(out0_ptr + out_cols, sum_xyhat, mask=out_mask)
        tl.store(out1_ptr + out_cols, sum_x, mask=out_mask)
        tl.store(out3_ptr + out_cols, sum_side, mask=out_mask)


def oracle_forward(inputs):
    """Compute the exact same four-output function as Repro.forward."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    mm_3, primals_158, addmm_49, getitem_141, rsqrt_25, select_scatter = inputs[:6]

    out0 = torch.empty((N_COLS,), device=mm_3.device, dtype=torch.float32)
    out1 = torch.empty((N_COLS,), device=mm_3.device, dtype=torch.float32)
    out2 = torch.empty((N_COLS, N_ROWS), device=mm_3.device, dtype=torch.float32)
    out3 = torch.empty((N_COLS,), device=mm_3.device, dtype=torch.float32)

    _side_transpose_kernel[(N_ROWS,)](
        mm_3,
        primals_158,
        addmm_49,
        getitem_141,
        rsqrt_25,
        select_scatter,
        out2,
        BLOCK_H=1024,
        num_warps=8,
    )
    _column_reduction_kernel[(triton.cdiv(N_COLS, 16),)](
        mm_3,
        addmm_49,
        getitem_141,
        rsqrt_25,
        out2,
        out0,
        out1,
        out3,
        BLOCK_M=128,
        BLOCK_C=16,
        num_warps=8,
    )
    return (out0, out1, out2, out3)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
