"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle returns the complete three-output Q/K/V split-view-permute result as metadata-only `as_strided` aliases of the f32[64,2304] input, whereas Inductor currently also eliminates the view/split/view/permute chain to alias metadata and CUDAGraph capture reports no executable GPU nodes for the oracle path; Inductor cannot provide a meaningful device-kernel floor today because scheduler/codegen receives no materialized producer after layout/view alias elimination; the fix is ALGEBRAIC_ELIMINATION: preserve this metadata-only rewrite and classify empty-graph timing as invalid/not a true floor rather than as a kernel optimization target."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@oracle_impl(hardware="H100", shapes="(T([64, 2304], f32), S([1, 64, 2304]), S([1, 64, 12, 64]), S([1, 64, 12, 64]), S([1, 64, 12, 64]))")
def oracle_forward(inputs):
    """Run the full-scope metadata-only QKV split/layout oracle.

    The captured chain contains only view, split, view, and permute operations,
    so the correct oracle result is three aliases of the original input storage
    with the exact output sizes, strides, and storage offsets.
    """
    x = inputs[0]
    view_shape = tuple(inputs[1])
    k_shape = tuple(inputs[2])
    q_shape = tuple(inputs[3])
    v_shape = tuple(inputs[4])

    view_stride = x.view(view_shape).stride()
    base_offset = x.storage_offset()

    def split_permute_view(chunk_index, head_shape):
        batch, seq, heads, head_dim = head_shape
        chunk_width = heads * head_dim
        size = (batch, heads, seq, head_dim)
        stride = (
            view_stride[0],
            head_dim * view_stride[2],
            view_stride[1],
            view_stride[2],
        )
        storage_offset = base_offset + chunk_index * chunk_width * view_stride[2]
        return torch.as_strided(x, size, stride, storage_offset=storage_offset)

    return (
        split_permute_view(1, k_shape),
        split_permute_view(0, q_shape),
        split_permute_view(2, v_shape),
    )


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
