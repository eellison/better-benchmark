"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bool-mask scale and metadata-only view scope in one storage-linear Triton pointwise kernel, preserving the eager NaN behavior of `mm_4 * (arg17_1.float() * 2.0)` and returning the same fresh contiguous buffer viewed as `_shape_param_0`, whereas Inductor already lowers this isolated convert/mul/mul/view graph to the same one-kernel pointwise memory-traffic envelope; Inductor cannot materially improve it today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the required bool/f32 input reads and f32 output write dominate the full captured computation; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise memory-bandwidth or launch-overhead work moves both paths."""
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bool_mul_kernel(
        mask_ptr,
        value_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < n_elements
        mask_values = tl.load(mask_ptr + offsets, mask=in_bounds, other=0).to(tl.float32)
        values = tl.load(value_ptr + offsets, mask=in_bounds, other=0.0)
        result = values * (mask_values * 2.0)
        tl.store(out_ptr + offsets, result, mask=in_bounds)


@oracle_impl(hardware="H100", shapes="(T([1024, 9216], b8), T([1024, 9216], f32), S([1024, 256, 6, 6]))")
def oracle_forward(inputs):
    """Run the full captured Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg17_1, mm_4, shape = inputs
    out_flat = torch.empty_strided(
        mm_4.shape,
        mm_4.stride(),
        device=mm_4.device,
        dtype=mm_4.dtype,
    )
    n_elements = mm_4.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _bool_mul_kernel[grid](
        arg17_1,
        mm_4,
        out_flat,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out_flat.view(shape)


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
