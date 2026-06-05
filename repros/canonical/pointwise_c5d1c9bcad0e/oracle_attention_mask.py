"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle materializes the complete six-output fp16 causal attention-mask tuple from the shape-derived iota/le predicate in one Triton pass, whereas Inductor lowers six identical where fills over the same expanded predicate as repeated generic pointwise work; Inductor cannot do this today because it does not canonicalize repeated shape-only causal-mask expressions into a shared multi-output materialization; the fix is ALGEBRAIC_ELIMINATION: recognize the iota/le lower-triangular mask, share the predicate, and emit one multi-output fill for all identical returned masks."""
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
ROWS = 512
COLS = 512
NUM_OUTPUTS = 6
OUTPUT_SHAPE = (1, 1, ROWS, COLS)
N_ELEMENTS = ROWS * COLS

if triton is not None:

    @triton.jit
    def oracle_kernel(
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        out4_ptr,
        out5_ptr,
        N: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Materialize six independent lower-triangular fp16 mask outputs."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        in_bounds = offsets < N
        row = offsets // COLS_
        col = offsets - row * COLS_
        values = tl.where(col <= row, 0.0, -float("inf"))

        tl.store(out0_ptr + offsets, values, mask=in_bounds)
        tl.store(out1_ptr + offsets, values, mask=in_bounds)
        tl.store(out2_ptr + offsets, values, mask=in_bounds)
        tl.store(out3_ptr + offsets, values, mask=in_bounds)
        tl.store(out4_ptr + offsets, values, mask=in_bounds)
        tl.store(out5_ptr + offsets, values, mask=in_bounds)


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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    if len(inputs) != 1:
        raise ValueError(f"expected one shape-param input, got {len(inputs)}")
    (shape_param,) = inputs
    if list(shape_param) != [1, -1, ROWS, COLS]:
        raise ValueError(f"unexpected shape parameter: {shape_param}")

    outputs = tuple(
        torch.empty(OUTPUT_SHAPE, device="cuda", dtype=torch.float16)
        for _ in range(NUM_OUTPUTS)
    )
    grid = (triton.cdiv(N_ELEMENTS, 256),)
    oracle_kernel[grid](
        *outputs,
        N=N_ELEMENTS,
        COLS_=COLS,
        BLOCK_N=256,
        num_warps=4,
    )
    return outputs


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
