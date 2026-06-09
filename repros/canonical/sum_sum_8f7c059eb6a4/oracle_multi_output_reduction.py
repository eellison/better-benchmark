"""
Oracle for sum_sum_8f7c059eb6a4

Gap diagnosis: Classification: SCHEDULER_FUSION. This oracle computes the full
return tuple in one Triton kernel by loading each two-element row once, forming
the row sum, writing the returned `[2, 128]` permuted `sub_tensor` layout
directly, and accumulating the returned `[2]` column-sum side output from the
same values, whereas Inductor schedules the row reduction, exp/multiply/sub
producer, transpose/layout output, and column reduction as separate generic
regions. Inductor cannot do this today because its scheduler does not represent
this small cross-axis multi-output reduction as one materializing producer plus
sibling reduction with a layout-changing side output. The Inductor fix
classification is SCHEDULER_FUSION: teach the scheduler/codegen to fuse
compatible row-local reductions, materialized side-output stores, and dependent
column reductions into one full-scope multi-output plan.
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

M = 128
N = 2
BLOCK_M = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _sum_sum_full_scope_kernel(
        arg285_ptr,
        arg336_ptr,
        out_perm_ptr,
        out_sum_ptr,
        BLOCK_M_: tl.constexpr,
    ):
        rows = tl.arange(0, BLOCK_M_)
        offsets0 = rows * 2
        offsets1 = offsets0 + 1

        a0 = tl.load(arg285_ptr + offsets0).to(tl.float32)
        a1 = tl.load(arg285_ptr + offsets1).to(tl.float32)
        b0 = tl.load(arg336_ptr + offsets0).to(tl.float32)
        b1 = tl.load(arg336_ptr + offsets1).to(tl.float32)

        row_sum = b0 + b1
        sub0 = b0 - tl.exp(a0) * row_sum
        sub1 = b1 - tl.exp(a1) * row_sum

        tl.store(out_perm_ptr + offsets0, sub0)
        tl.store(out_perm_ptr + offsets1, sub1)
        tl.store(out_sum_ptr + 0, tl.sum(sub0, axis=0))
        tl.store(out_sum_ptr + 1, tl.sum(sub1, axis=0))


@oracle_impl(hardware="H100", shapes="(T([128, 2], f32), T([128, 2], f32), S([2]))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    if len(inputs) != 3:
        raise RuntimeError(f"expected three inputs, got {len(inputs)}")
    arg285_1, arg336_1, shape_param = inputs

    if not isinstance(arg285_1, torch.Tensor) or not isinstance(arg336_1, torch.Tensor):
        raise RuntimeError("expected tensor inputs for arg285_1 and arg336_1")
    if arg285_1.shape != (M, N) or arg336_1.shape != (M, N):
        raise RuntimeError(f"expected [128, 2] inputs, got {arg285_1.shape} and {arg336_1.shape}")
    if arg285_1.dtype != torch.float32 or arg336_1.dtype != torch.float32:
        raise RuntimeError(f"expected float32 inputs, got {arg285_1.dtype} and {arg336_1.dtype}")
    if not arg285_1.is_cuda or not arg336_1.is_cuda:
        raise RuntimeError("CUDA tensors are required")
    if not arg285_1.is_contiguous() or not arg336_1.is_contiguous():
        raise RuntimeError("this oracle expects contiguous [128, 2] inputs")

    out_perm = torch.empty_strided(
        (N, M),
        (1, N),
        device=arg336_1.device,
        dtype=arg336_1.dtype,
    )
    out_sum = torch.empty(tuple(int(x) for x in shape_param), device=arg336_1.device, dtype=arg336_1.dtype)
    _sum_sum_full_scope_kernel[(1,)](
        arg285_1,
        arg336_1,
        out_perm,
        out_sum,
        BLOCK_M_=BLOCK_M,
        num_warps=4,
    )
    return out_perm, out_sum


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
