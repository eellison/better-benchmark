"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet f32 [512,1280,1,1] ReLU and returns the flattened [512,1280] contiguous view in one streaming Triton pointwise kernel, whereas Inductor already lowers this repro to the same essential one-read/one-write fused pointwise structure with the reshape folded as metadata; Inductor cannot materially improve this today because the captured computation is dominated by the required input read, output write, and launch overhead with no removable intermediate or missing fusion left in scope; the fix is BANDWIDTH_BOUND: record this as a pointwise memory-bandwidth floor unless broader pointwise scheduler/codegen overhead reductions move the baseline."""
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

ROWS = 512
COLS = 1280
INPUT_SHAPE = (ROWS, COLS, 1, 1)
OUTPUT_SHAPE = (ROWS, COLS)
OUTPUT_STRIDE = (COLS, 1)
N_ELEMENTS = ROWS * COLS
BLOCK_SIZE = 1024

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


if triton is not None:

    @triton.jit
    def _relu_flatten_kernel(
        input_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(x != x, x, tl.maximum(x, 0.0))
        tl.store(output_ptr + offsets, y, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    convolution_94, shape_param = inputs
    if not isinstance(convolution_94, torch.Tensor):
        raise TypeError("expected first input to be a tensor")
    if convolution_94.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if convolution_94.dtype != torch.float32:
        raise TypeError(f"expected f32 input, got {convolution_94.dtype}")
    if tuple(convolution_94.shape) != INPUT_SHAPE:
        raise ValueError(f"expected input shape {INPUT_SHAPE}, got {tuple(convolution_94.shape)}")
    if not convolution_94.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layout")
    if tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"expected reshape target {OUTPUT_SHAPE}, got {tuple(shape_param)}")
    return convolution_94


@oracle_impl(hardware="H100", shapes="(T([512, 1280, 1, 1], f32), S([512, 1280]))")
def oracle_forward(inputs):
    """Run the full Repro.forward ReLU plus flatten scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    convolution_94 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=convolution_94.device,
        dtype=convolution_94.dtype,
    )
    grid = (triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)
    _relu_flatten_kernel[grid](
        convolution_94,
        output,
        n_elements=N_ELEMENTS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
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
