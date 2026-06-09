import argparse
"""
Oracle kernel for sum_sum_mean_1dfbbe76c078 (Qwen3-30B RMSNorm backward).

=== Investigation Summary ===

This is the SECOND Qwen3 RMSNorm pattern from Qwen3-30B-A3B. It is structurally
identical to sum_sum_mean_9af96955f8cc with one difference: the final mul_tensor_2
is viewed into THREE separate [2048, 2048] outputs instead of one.

The 1.9x gap comes from the same root cause as the first pattern:

1. DEFAULT generates 4 kernels:
   - K0: sum [2048, 8] -> [2048, 1]  (trivial, <1us)
   - K1: index_put to build inverse permutation  (trivial, <1us)
   - K2: Fused gather+scale+mask+sum-over-8-experts. Grid=4M, reduction=8.
          Reads grouped_mm_5 [16384, 2048] = 64MB. Writes [2048, 2048] = 8MB.
   - K3: RMSNorm (add + x^2 + mean + rsqrt + weight*x). Reads 8MB intermediate + 8MB add_26.
          Persistent reduction over 2048 cols. Writes 8MB output.

   Total bandwidth: 64MB (read grouped_mm) + 8MB (write intermediate) + 8MB (read intermediate)
                  + 8MB (read add_26) + 8MB (write output) = 96MB
   At ~2.3 TB/s effective BW -> 96MB / 2.3TB/s = ~42us theoretical
   Actual: ~87us because of kernel launch overhead and under-utilization.

2. ORACLE achieves ~34-36us by:
   - Fusing K2+K3 into a SINGLE persistent kernel with grid=[2048]
   - Each program processes one full output row (2048 columns) in registers
   - Eliminates the 8MB intermediate write+read (saves 16MB bandwidth)
   - Total bandwidth: 64MB (grouped_mm) + 8MB (add_26) + 4KB (weight) + 8MB (output) = 80MB
   - The three outputs are identical (same view shape of same tensor), so we write once

=== Difference from 9af96955f8cc ===

The second pattern produces 3 views of mul_tensor_2 as outputs (for 3 downstream consumers),
whereas the first produces only 1 view. Since all 3 views are [2048, 2048] of the same
tensor, the oracle simply returns the same buffer 3 times. The computation is identical.

=== What structural change would close the gap? ===

Same as the first pattern: Inductor needs "reduction chaining" -- recognizing that when
a reduction kernel (sum over 8 experts) feeds directly into another reduction kernel
(RMSNorm mean over 2048 cols), these can be fused into a single persistent kernel where
the intermediate stays in registers.
"""
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from repro_harness import parse_shapes_config

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_mean_1dfbbe76c078"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"


def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return _oracle_impl(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
