"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the bool-to-float scale, dropout where, full base materialization, and column reduction while returning the transposed view of the same base, whereas Inductor emits separate work for the materialized transpose-producing tensor and the reduction; Inductor cannot do this today because the scheduler does not fuse a producer that must be both materialized for a layout-changing view and reduced across another consumer; the fix is SCHEDULER_FUSION: allow multi-output scheduling that stores the shared producer once and computes the reduction in the same pointwise/reduction kernel."""
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 16, "RBLOCK": 32}, num_warps=4, num_stages=2),
            triton.Config({"XBLOCK": 32, "RBLOCK": 16}, num_warps=4, num_stages=2),
            triton.Config({"XBLOCK": 32, "RBLOCK": 32}, num_warps=4, num_stages=2),
            triton.Config({"XBLOCK": 64, "RBLOCK": 16}, num_warps=4, num_stages=2),
            triton.Config({"XBLOCK": 64, "RBLOCK": 32}, num_warps=4, num_stages=2),
            triton.Config({"XBLOCK": 128, "RBLOCK": 16}, num_warps=4, num_stages=2),
        ],
        key=["N"],  # re-tune when N changes
    )
    @triton.jit
    def oracle_kernel(
        scale_mask_ptr,
        mm_ptr,
        zero_mask_ptr,
        base_out_ptr,
        sum_out_ptr,
        N: tl.constexpr,
        M: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        """Materialize the full masked product and reduce columns."""
        pid_n = tl.program_id(0)
        cols = pid_n * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        rbase = tl.arange(0, RBLOCK)[None, :]
        acc = tl.zeros((XBLOCK, RBLOCK), tl.float32)

        for roff in tl.range(0, M, RBLOCK):
            rows = roff + rbase
            offsets = rows * N + cols
            scale_mask = tl.load(scale_mask_ptr + offsets).to(tl.float32)
            mm = tl.load(mm_ptr + offsets)
            zero_mask = tl.load(zero_mask_ptr + offsets)
            values = tl.where(zero_mask, 0.0, mm * scale_mask * 2.0)
            tl.store(base_out_ptr + offsets, values)
            acc += values

        sums = tl.sum(acc, axis=1)[:, None]
        tl.store(sum_out_ptr + cols, sums)


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
    arg38_1, mm, arg40_1, shape_param_0 = inputs
    m, n = mm.shape
    base = torch.empty_like(mm)
    summed = torch.empty((n,), device=mm.device, dtype=mm.dtype)
    grid = lambda meta: (triton.cdiv(n, meta["XBLOCK"]),)
    oracle_kernel[grid](
        arg38_1,
        mm,
        arg40_1,
        base,
        summed,
        N=n,
        M=m,
    )
    return base.permute(1, 0), summed.view(shape_param_0)


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
