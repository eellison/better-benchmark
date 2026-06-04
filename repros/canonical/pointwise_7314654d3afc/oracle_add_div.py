"""
Oracle for pointwise_7314654d3afc

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
`Repro.forward` scope with one Triton pointwise kernel that reads both
`float32[128,1000]` inputs, performs the `aten.add.Tensor` in float32, then
performs the `aten.div.Tensor(..., 2)` in float32 and writes the contiguous
`float32[128,1000]` result, whereas Inductor's best lowering for this repro is
also a single fused pointwise add/div kernel over the same two reads and one
write; Inductor cannot materially improve this today because the remaining cost
is launch overhead plus mandatory memory traffic rather than an avoidable
materialization or missed fusion; the fix is BANDWIDTH_BOUND: record the
full-scope floor and only reopen for broad pointwise launch-overhead or memory
codegen improvements.
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _add_div_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        summed = x + y
        out = summed / 2.0
        tl.store(out_ptr + offsets, out, mask=mask)


def oracle_forward(inputs):
    """Run the full add-then-div repro computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm_48, addmm_49 = inputs
    out = torch.empty_like(addmm_48)
    n_elements = out.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _add_div_kernel[grid](
        addmm_48,
        addmm_49,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify correctness against eager Repro"
    )
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument(
        "--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check"
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt"
    )
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
                        "WARNING: oracle is slower than compile for "
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
