"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full captured MT5 attention-mask bias construction returned by Repro.forward, including the tautological iota(128) >= 0 predicate, broadcast where-to-zero, add with a zero [1,6,128,128] bias, and eight aliasing expanded [32,6,128,128] outputs, whereas Inductor currently lowers the decomposed full/iota/add/unsqueeze/ge/expand/where/add/expand graph as generic pointwise mask construction and repeated metadata outputs; Inductor cannot do this today because its simplifier does not prove the shape-derived iota predicate is always true through unsqueeze/expand/where and fold the zero broadcast before scheduling; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate and zero-bias canonicalization for structured attention masks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

MASK_EXPAND_SHAPE = (32, -1, 128, 128)
BIAS_SHAPE = (32, 6, 128, 128)
BIAS_STRIDE = (6 * 128 * 128, 128 * 128, 128, 1)
OUTPUT_COUNT = 8

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
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[tuple[int, ...], ...]:
    if len(inputs) != OUTPUT_COUNT + 1:
        raise ValueError(f"{REPRO_ID} expects {OUTPUT_COUNT + 1} shape inputs, got {len(inputs)}")

    mask_shape = _shape_tuple(inputs[0])
    if mask_shape != MASK_EXPAND_SHAPE:
        raise ValueError(
            f"unexpected mask expand shape parameter {mask_shape}, expected {MASK_EXPAND_SHAPE}"
        )

    output_shapes = tuple(_shape_tuple(shape) for shape in inputs[1:])
    for index, shape in enumerate(output_shapes, start=1):
        if shape != BIAS_SHAPE:
            raise ValueError(f"unexpected output expand shape {index}: {shape}, expected {BIAS_SHAPE}")

    return output_shapes


@oracle_impl(hardware="H100", shapes="(S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward MT5 attention-mask bias construction.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same eight fp32 [32,6,128,128] outputs. The returned tensors are
    metadata-only expands over the same full zero bias buffer, matching the
    eager repeated-expand alias contract.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    output_shapes = _validate_inputs(inputs)
    bias = torch.empty_strided(
        BIAS_SHAPE,
        BIAS_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    bias.zero_()
    return tuple(bias.expand(shape) for shape in output_shapes)


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
