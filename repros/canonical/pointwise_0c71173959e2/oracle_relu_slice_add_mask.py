"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs relu/slice/add/mask scope in one flat Triton pointwise kernel, including the full `relu(convolution_9) + arg26_1[:, :, 1426:-1426]` float output and the boolean `relu(convolution_9) <= 0` output, while folding the mask to the equivalent `convolution_9 <= 0` predicate and sharing the single convolution load; Inductor already lowers the full scope to one fused pointwise kernel that reads both inputs and stores both outputs, retaining only a non-material compare against the relu value; Inductor cannot materially improve this today through local scheduler fusion, scatter/reduce, split-K, algebraic elimination, or recomputation because runtime is dominated by the mandatory multi-GB pointwise memory traffic, so the fix is BANDWIDTH_BOUND: record this repro as an at-floor case unless broader pointwise memory-codegen or launch/allocator work moves both implementations together."""
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N", "M"],
    )
    @triton.jit
    def _relu_slice_add_mask_kernel(
        convolution_ptr,
        arg26_ptr,
        add_out_ptr,
        mask_out_ptr,
        N: tl.constexpr,
        M: tl.constexpr,
        SLICE_START: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        cols = offsets % N
        rows = offsets // N
        arg_offsets = rows * M + SLICE_START + cols

        conv = tl.load(convolution_ptr + offsets)
        sliced = tl.load(arg26_ptr + arg_offsets)
        relu = tl.maximum(conv, 0.0)

        tl.store(add_out_ptr + offsets, relu + sliced)
        tl.store(mask_out_ptr + offsets, conv <= 0.0)


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    convolution_9, arg26_1 = inputs
    if not isinstance(convolution_9, torch.Tensor) or not isinstance(arg26_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor inputs")
    if not convolution_9.is_cuda or not arg26_1.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA tensors")
    if convolution_9.dtype is not torch.float32 or arg26_1.dtype is not torch.float32:
        raise ValueError(
            f"{REPRO_ID} expects float32 inputs, got {convolution_9.dtype} and {arg26_1.dtype}"
        )
    if convolution_9.ndim != 3 or arg26_1.ndim != 3:
        raise ValueError(
            f"{REPRO_ID} expects rank-3 inputs, got {tuple(convolution_9.shape)} "
            f"and {tuple(arg26_1.shape)}"
        )
    if tuple(convolution_9.shape[:2]) != tuple(arg26_1.shape[:2]):
        raise ValueError(
            f"{REPRO_ID} expects matching leading dimensions, got "
            f"{tuple(convolution_9.shape)} and {tuple(arg26_1.shape)}"
        )
    if not convolution_9.is_contiguous() or not arg26_1.is_contiguous():
        raise ValueError(
            f"{REPRO_ID} expects contiguous inputs, got strides "
            f"{tuple(convolution_9.stride())} and {tuple(arg26_1.stride())}"
        )

    n = int(convolution_9.shape[2])
    m = int(arg26_1.shape[2])
    slice_start = 1426
    if m - 2 * slice_start != n:
        raise ValueError(
            f"{REPRO_ID} expects arg26_1 dim2 to exceed convolution dim2 by "
            f"{2 * slice_start}, got {m} and {n}"
        )

    rows = int(convolution_9.shape[0] * convolution_9.shape[1])
    return convolution_9, arg26_1, rows, n, m, slice_start


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

    convolution_9, arg26_1, rows, n, m, slice_start = _validate_inputs(inputs)
    add_out = torch.empty_strided(
        tuple(convolution_9.shape),
        tuple(convolution_9.stride()),
        device=convolution_9.device,
        dtype=convolution_9.dtype,
    )
    mask_out = torch.empty_strided(
        tuple(convolution_9.shape),
        tuple(convolution_9.stride()),
        device=convolution_9.device,
        dtype=torch.bool,
    )

    total = int(rows * n)
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    _relu_slice_add_mask_kernel[grid](
        convolution_9,
        arg26_1,
        add_out,
        mask_out,
        N=n,
        M=m,
        SLICE_START=slice_start,
    )
    return add_out, mask_out


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
