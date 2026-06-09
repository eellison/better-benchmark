"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full view-add-add-permute-add expression in one contiguous-output Triton pointwise kernel, whereas Inductor should already lower this pure pointwise view/permute chain to the same single fused memory-streaming computation; Inductor cannot materially beat this today because the operation has no reusable arithmetic and must read four f32 tensors and write one f32 tensor with a strided access pattern forced by the requested layout; the fix is BANDWIDTH_BOUND: no compiler transform beyond preserving single-kernel pointwise fusion and avoiding extra materialization is expected to move the floor."""
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


if triton is not None:

    @triton.jit
    def _pointwise_kernel(
        mm_139,
        mm_141,
        mm_143,
        mul_305,
        out,
        B: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h < H

        s = row % S
        b = row // S
        out_offsets = row * H + h
        mm_offsets = (s * B + b) * H + h
        values = (
            tl.load(mul_305 + out_offsets, mask=mask)
            + tl.load(mm_139 + mm_offsets, mask=mask)
            + tl.load(mm_141 + mm_offsets, mask=mask)
            + tl.load(mm_143 + mm_offsets, mask=mask)
        )
        tl.store(out + out_offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8, 1024, 768], f32), S([1024, 8, 768]), S([1024, 8, 768]), S([1024, 8, 768]))")
def oracle_forward(inputs):
    """Run the full Repro() computation for the same inputs and output layout."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    mm_139, mm_141, mm_143, mul_305, shape0, shape1, shape2 = inputs
    if shape0 != shape1 or shape0 != shape2:
        raise RuntimeError("Expected all view shape parameters to match")

    bsz, seq, hidden = mul_305.shape
    if tuple(shape0) != (seq, bsz, hidden):
        raise RuntimeError(
            f"Unexpected view shape {shape0}; expected {(seq, bsz, hidden)}"
        )

    out = torch.empty_strided(
        mul_305.shape,
        mul_305.stride(),
        device=mul_305.device,
        dtype=mul_305.dtype,
    )
    block_h = 1024
    grid = (bsz * seq, triton.cdiv(hidden, block_h))
    _pointwise_kernel[grid](
        mm_139,
        mm_141,
        mm_143,
        mul_305,
        out,
        bsz,
        seq,
        hidden,
        BLOCK_H=block_h,
    )
    return out


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
