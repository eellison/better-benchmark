"""
Oracle for pointwise_d9f9b7733ca3

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the exact
no-input repro by allocating the returned contiguous `float32[16, 1, 1, 512]`
layout and materializing `-0.0` with one Triton kernel that writes the f32
negative-zero bit pattern, whereas Inductor already lowers the graph to
equivalent tiny constant-fill work for the returned tensor. Inductor cannot
eliminate this work today without changing the eager contract because the
fresh returned tensor must be allocated and its observable values must be
materialized; the fix is BANDWIDTH_BOUND: only generic launch/allocation
overhead or caller-owned-output reuse would move this floor, not a new fusion,
scatter, split-K, algebraic, or recomputation pattern.
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


OUT_SHAPE = (16, 1, 1, 512)
OUT_STRIDE = (512, 512, 512, 1)
OUT_NUMEL = 16 * 1 * 1 * 512
FILL_BLOCK = 512
NEG_ZERO_F32_BITS = 0x80000000


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fill_neg_zero_bits_kernel(
        out_bits_ptr,
        neg_zero_bits: tl.constexpr,
        numel: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < numel
        values = tl.full((block,), neg_zero_bits, tl.uint32)
        tl.store(out_bits_ptr + offsets, values, mask=mask)


def oracle_forward(inputs):
    """Materialize the same f32 negative-zero tensor returned by Repro.forward()."""
    if len(inputs) != 0:
        raise AssertionError(f"expected no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    _fill_neg_zero_bits_kernel[(triton.cdiv(OUT_NUMEL, FILL_BLOCK),)](
        output.view(torch.uint32),
        neg_zero_bits=NEG_ZERO_F32_BITS,
        numel=OUT_NUMEL,
        block=FILL_BLOCK,
        num_warps=4,
    )
    if tuple(output.shape) != OUT_SHAPE or output.stride() != OUT_STRIDE:
        raise AssertionError(
            f"oracle layout mismatch: shape={tuple(output.shape)} stride={output.stride()}"
        )
    return output


def _check_exact_layout_and_bits(instance, inputs):
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == OUT_SHAPE
        and oracle_out.stride() == OUT_STRIDE
        and oracle_out.dtype == torch.float32
    )
    bits_ok = torch.equal(
        eager_out.view(torch.uint32),
        oracle_out.view(torch.uint32),
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} dtype={oracle_out.dtype})"
    )
    print(f"  output 0 bit_pattern: {'PASS' if bits_ok else 'FAIL'}")
    return layout_ok and bits_ok


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
        ok = ok and _check_exact_layout_and_bits(instance, inputs)
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
