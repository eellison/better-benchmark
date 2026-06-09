"""
Oracle for sum_df75f19c9c7a

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete Repro.forward return in one Triton kernel by evaluating the shared
`mm_5 * (1 - arg5_1 * arg5_1)` producer once, writing the full permuted
`[16, 1000]` materialization with eager stride `(1, 16)`, and accumulating the
live `[16]` column sum while each column's values are still in registers.
Inductor currently lowers the materialized permute side output and the
reduction output as separate scheduled consumers of the same pointwise
producer; it cannot do this today because the generic scheduler has no
materializing multi-output reduction template that combines a layout-only side
store with a sibling reduction over that store's source domain. The fix is
SCHEDULER_FUSION: teach Inductor to fuse shared pointwise producers into both
required layout stores and small reductions when all consumers are inside the
captured graph.
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N_ROWS = 1000
N_COLS = 16
BLOCK_ROWS = 1024
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _permuted_materialize_and_sum_kernel(
        arg5_ptr,
        mm5_ptr,
        permuted_out_ptr,
        sum_out_ptr,
        BLOCK_M: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
    ):
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        mask = rows < N
        offsets = rows * C + col

        arg5 = tl.load(arg5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mm5 = tl.load(mm5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        value = mm5 * (1.0 - arg5 * arg5)

        # Output 0 is aten.permute(value, [1, 0]); its eager logical stride is
        # (1, 16), so source row-major offsets are also output storage offsets.
        tl.store(permuted_out_ptr + offsets, value, mask=mask)
        tl.store(sum_out_ptr + col, tl.sum(value, axis=0))


def _validate_inputs(inputs) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    arg5_1, mm_5, shape = inputs
    if not (arg5_1.is_cuda and mm_5.is_cuda):
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if arg5_1.shape != (N_ROWS, N_COLS) or arg5_1.dtype != torch.float32:
        raise ValueError(f"expected arg5_1 f32[{N_ROWS}, {N_COLS}]")
    if mm_5.shape != (N_ROWS, N_COLS) or mm_5.dtype != torch.float32:
        raise ValueError(f"expected mm_5 f32[{N_ROWS}, {N_COLS}]")
    if not (arg5_1.is_contiguous() and mm_5.is_contiguous()):
        raise ValueError("oracle expects the canonical contiguous repro inputs")
    if tuple(int(dim) for dim in shape) != (N_COLS,):
        raise ValueError(f"expected _shape_param_0 [{N_COLS}], got {shape}")


@oracle_impl(hardware="H100", shapes="(T([1000, 16], f32), T([1000, 16], f32), S([16]))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation for both repro outputs."""
    _validate_inputs(inputs)
    arg5_1, mm_5, _shape_param_0 = inputs

    permuted = torch.empty_strided(
        (N_COLS, N_ROWS),
        (1, N_COLS),
        device=arg5_1.device,
        dtype=arg5_1.dtype,
    )
    reduced = torch.empty(tuple(_shape_param_0), device=arg5_1.device, dtype=arg5_1.dtype)

    _permuted_materialize_and_sum_kernel[(N_COLS,)](
        arg5_1,
        mm_5,
        permuted,
        reduced,
        BLOCK_M=BLOCK_ROWS,
        N=N_ROWS,
        C=N_COLS,
        num_warps=8,
    )
    return permuted, reduced


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
