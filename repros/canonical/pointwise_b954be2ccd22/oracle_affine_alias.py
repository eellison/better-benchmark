"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full MobileBERT affine pointwise scope in one hand-written Triton pass and returns the three `[32768, 512]` outputs as metadata views aliasing one computed buffer, whereas Inductor lowers the same add/mul/add/view graph through its generic pointwise scheduler and already preserves the single-buffer aliasing; Inductor cannot materially improve this local repro today because the remaining work is the mandatory two activation reads, two broadcast parameter reads, one output store, and launch overhead rather than a missed fusion, scatter-reduce, split-K, algebraic-elimination, or recompute opportunity; the fix is BANDWIDTH_BOUND: record this as at floor unless broader pointwise bandwidth or launch-overhead work moves both implementations."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _affine_alias_kernel(
        addmm_ptr,
        add_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        cols = offsets % n_cols

        x = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(add_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        out = (x + residual) * scale + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward scope and preserve the three aliasing views."""
    addmm_345, add_333, arg1065_1, arg1066_1 = inputs[:4]
    shape_1, shape_2, shape_3 = inputs[5], inputs[6], inputs[7]

    if triton is None or not addmm_345.is_cuda:
        computed = (addmm_345.view(inputs[4]) + add_333) * arg1065_1 + arg1066_1
        return (computed.view(shape_1), computed.view(shape_2), computed.view(shape_3))

    out = torch.empty_strided(
        tuple(shape_1),
        (shape_1[1], 1),
        device=addmm_345.device,
        dtype=addmm_345.dtype,
    )
    n_elements = addmm_345.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _affine_alias_kernel[grid](
        addmm_345,
        add_333,
        arg1065_1,
        arg1066_1,
        out,
        n_elements,
        addmm_345.shape[1],
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return (out.view(shape_1), out.view(shape_2), out.view(shape_3))


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
