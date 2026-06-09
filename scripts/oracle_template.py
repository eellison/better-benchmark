"""Gap diagnosis (classification: <SCHEDULER_FUSION|SCATTER_REDUCE|COOPERATIVE_SPLIT_K|ALGEBRAIC_ELIMINATION|NEW_PATTERN>): this oracle <specific behavior>, whereas Inductor <specific current behavior>; Inductor cannot do this today because <specific scheduler/codegen/pattern limitation>; the fix is <CLASS>: <specific Inductor change>."""
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
    oracle_impl,  # standard: declare what hw/shapes this oracle was written for
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
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=2),
        ],
        key=["N"],  # re-tune when N changes
    )
    @triton.jit
    def oracle_kernel(
        # TODO: Define kernel parameters
        # input_ptr, output_ptr, ...
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """TODO: Implement optimized kernel."""
        pass


# STANDARD: declare what this oracle was written for — one line. Copy the
# `shapes` string verbatim from the repro's `_shapes_config`. It documents the
# exact operating point (full input signature) and hardware this kernel was
# tuned on; bench output then reports dispatch tier / fallback honestly
# (e.g. an H100-tuned kernel measured on B200 reports tier 2, fallback=true).
#
# Matching is exact-only: this impl runs ONLY at its declared signature.
# If the impl is genuinely shape-general (grid computed from input dims,
# autotuned), register it with shapes=None instead — it then serves any
# shape. To tune several shapes.txt lines, add one registration per line
# (same body, different configs).
@oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")  # <- _shapes_config
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
    # TODO: Implement oracle using the Triton kernel(s) above.
    # Example:
    #   x = inputs[0]
    #   output = torch.empty_like(x)
    #   grid = (x.shape[0],)
    #   oracle_kernel[grid](x, output, N=x.shape[1], BLOCK_N=8192)
    #   return output
    raise NotImplementedError("Replace with oracle implementation")


# --- OPTIONAL: variants for other hardware / shapes ---
# When an oracle's configs don't transfer (e.g. H100-tuned BLOCK sizes are
# BAD_ORACLE on B200), add a variant instead of forking the file. One kernel
# body can be registered N times with different launch configs — `configs`
# is passed to the function as kwargs at dispatch:
#
# def _softmax_impl(inputs, *, BLOCK, num_warps):
#     x = inputs[0]
#     out = torch.empty_like(x)
#     oracle_kernel[(x.shape[0],)](x, out, N=x.shape[1],
#                                  BLOCK_N=BLOCK, num_warps=num_warps)
#     return out
#
# SIG = "(T([32768, 1024], bf16),)"  # _shapes_config, shared across variants
# oracle_impl(hardware="H100", shapes=SIG, configs={"BLOCK": 1024, "num_warps": 4})(_softmax_impl)
# oracle_impl(hardware="B200", shapes=SIG, configs={"BLOCK": 2048, "num_warps": 8})(_softmax_impl)
#
# A genuinely different algorithm is just another decorated function:
#
# @oracle_impl(hardware="B200", shapes="(T([8192, 262144], bf16),)",
#              description="split-K two-pass for huge inner dim")
# def _oracle_split_k(inputs): ...
#
# Match order: "hardware+shape" (exact sig + this GPU) > "shape" (exact
# sig, tuned elsewhere) > "hardware" (shape-general, this GPU) > "any"
# (shape-general, unconstrained). No match -> NO_ORACLE_FOR_SHAPE in bench
# output — never a silent wrong-shape run. dtypes in the signature are
# documentation; matching is shape-only (the corpus dedupes across dtypes).


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
