"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full OPT/vLLM addmm-scale-view-permute scope by scaling the contiguous f16 `[2048, 768]` storage once and returning the exact non-contiguous `[4, 12, 512, 64]` view with stride `(393216, 64, 768, 1)`, whereas tuned Inductor reaches the same one-kernel storage-linear scale/copy envelope for this metadata-only layout chain; Inductor cannot materially improve this local region today because the remaining cost is the required fresh output allocation, f16 input read, f16 output store, and launch overhead rather than an avoidable intermediate or indexing pattern; the fix is BANDWIDTH_BOUND: record this layout-indexing stencil-fusion repro as at floor unless a broader allocation or launch-overhead optimization changes the baseline."""
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


INPUT_SHAPE = (2048, 768)
SHAPE_PARAM_0 = [4, 512, 768]
SHAPE_PARAM_1 = [4, -1, 12, 64]
OUTPUT_SHAPE = (4, 12, 512, 64)
OUTPUT_STRIDE = (393216, 64, 768, 1)
NUMEL = 2048 * 768
BLOCK_N = 8192


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

    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Scale the underlying contiguous storage for the permuted output view."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, values * 0.125, mask=mask)


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
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    addmm, shape_param_0, shape_param_1 = inputs
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if tuple(addmm.shape) != INPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} expects input shape {INPUT_SHAPE}, got {tuple(addmm.shape)}")
    if addmm.dtype != torch.float16:
        raise ValueError(f"{REPRO_ID} expects torch.float16 input, got {addmm.dtype}")
    if not addmm.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if not addmm.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={addmm.stride()}")
    if list(shape_param_0) != SHAPE_PARAM_0:
        raise ValueError(f"{REPRO_ID} input 1 must be {SHAPE_PARAM_0}, got {shape_param_0}")
    if list(shape_param_1) != SHAPE_PARAM_1:
        raise ValueError(f"{REPRO_ID} input 2 must be {SHAPE_PARAM_1}, got {shape_param_1}")

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=addmm.dtype,
    )
    oracle_kernel[(NUMEL // BLOCK_N,)](
        addmm,
        output,
        N=NUMEL,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return output


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
