"""
Full-scope oracle for pointwise_eba0147d8378.

Gap diagnosis (classification: BANDWIDTH_BOUND): The oracle materializes the exact no-input `full(-0.0)` storage with one Triton negative-zero fill kernel and returns the same two-selected `float32[8, 1024]` view as the repro, while Inductor also has to allocate and materialize a fresh CUDA output for this tiny constant-select graph. Inductor cannot make this materially faster today without avoiding the required launch/allocation/output-store work or changing the fresh-tensor semantics; the Inductor-side resolution is to classify this as a launch/materialization floor rather than a missing compiler transformation.
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BASE_SHAPE = (8, 1, 1, 1024)
BASE_STRIDE = (1024, 1024, 1024, 1)
OUT_SHAPE = (8, 1024)
OUT_STRIDE = (1024, 1)
OUT_NUMEL = 8 * 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fill_neg_zero_kernel(
        out_ptr,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        values = tl.full((BLOCK,), -2147483648, tl.int32).to(tl.float32, bitcast=True)
        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs):
    """Run the full repro computation and return the selected constant view."""
    if len(inputs) != 0:
        raise AssertionError(f"expected no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    block = 1024
    _fill_neg_zero_kernel[(triton.cdiv(OUT_NUMEL, block),)](
        base,
        NUMEL=OUT_NUMEL,
        BLOCK=block,
        num_warps=4,
    )
    out = base.select(1, 0).select(1, 0)
    if tuple(out.shape) != OUT_SHAPE or out.stride() != OUT_STRIDE:
        raise AssertionError(
            f"oracle layout mismatch: shape={tuple(out.shape)} stride={out.stride()}"
        )
    return out


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
        with torch.no_grad():
            eager_out = instance(*inputs)
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = (
            tuple(layout_out.shape) == OUT_SHAPE
            and layout_out.stride() == OUT_STRIDE
            and layout_out.storage_offset() == 0
        )
        sign_ok = bool(torch.signbit(eager_out).all() and torch.signbit(layout_out).all())
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()})"
        )
        print(f"  output 0 negative zero: {'PASS' if sign_ok else 'FAIL'}")
        ok = ok and layout_ok and sign_ok
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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
