"""
Oracle for sum_7c50e5b80bfe

Gap diagnosis: Classification: SCHEDULER_FUSION. The oracle uses one Triton
kernel to compute both column producers, store the full permuted [2, 256]
materialization, and accumulate both column sums while the values are still in
registers; Inductor cannot do this today because the scheduler treats the
permuted cat materialization and the sibling reduction as separate consumers
that are not fused into one producer/reduction loop nest, so the required
Inductor change is scheduler fusion for a shared pointwise producer with a
materialized side output and a small reduction output.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _multi_output_reduction_kernel(
        arg8_ptr,
        arg9_ptr,
        arg7_ptr,
        arg10_ptr,
        arg5_ptr,
        arg6_ptr,
        permute_out_ptr,
        sum_out_ptr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_N)

        col0 = tl.load(arg7_ptr + offsets) + tl.load(arg10_ptr + offsets)
        col1 = (
            (tl.load(arg8_ptr + offsets) + tl.load(arg9_ptr + offsets))
            * tl.load(arg5_ptr + offsets)
            * 6.0
            * tl.load(arg6_ptr + offsets)
        )

        # Output 0 is aten.permute(cat([col0, col1], 1), [1, 0]) with stride
        # (1, 2), so linear storage is the original contiguous cat layout.
        storage_offsets = offsets * 2
        tl.store(permute_out_ptr + storage_offsets, col0)
        tl.store(permute_out_ptr + storage_offsets + 1, col1)

        tl.store(sum_out_ptr + 0, tl.sum(col0, axis=0))
        tl.store(sum_out_ptr + 1, tl.sum(col1, axis=0))


def _torch_fallback(inputs):
    arg8_1, arg9_1, arg7_1, arg10_1, arg5_1, arg6_1, shape = inputs
    col0 = arg7_1 + arg10_1
    col1 = (arg8_1 + arg9_1) * arg5_1 * 6.0 * arg6_1
    cat = torch.cat([col0, col1], 1)
    return cat.permute(1, 0), cat.sum(dim=[0], keepdim=True).view(shape)


@oracle_impl(hardware="H100", shapes="(T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), T([256, 1], f32), S([2]))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation for both repro outputs."""
    if triton is None:
        return _torch_fallback(inputs)

    arg8_1, arg9_1, arg7_1, arg10_1, arg5_1, arg6_1, _shape_param_0 = inputs
    n_rows = arg8_1.numel()
    if n_rows != 256:
        return _torch_fallback(inputs)

    permute_out = torch.empty_strided(
        (2, 256),
        (1, 2),
        device=arg8_1.device,
        dtype=arg8_1.dtype,
    )
    sum_out = torch.empty((2,), device=arg8_1.device, dtype=arg8_1.dtype)

    _multi_output_reduction_kernel[(1,)](
        arg8_1,
        arg9_1,
        arg7_1,
        arg10_1,
        arg5_1,
        arg6_1,
        permute_out,
        sum_out,
        BLOCK_N=256,
    )
    return permute_out, sum_out


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
