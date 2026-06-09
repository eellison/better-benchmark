"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete LLaMA SwiGLU pointwise scope as one shape-specialized Triton pass over the contiguous f32 inputs, folding the metadata views and writing the returned f32[1024,1536] tensor directly, whereas tuned Inductor already lowers this isolated `x / (1 + exp(-x)) * gate` region to the same single fused pointwise-kernel envelope; Inductor cannot materially improve this repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or a new semantic pattern because the remaining work is the required two f32 input reads, exp/div/mul math, one f32 output store, and one CUDA-graph replayed launch; the fix is BANDWIDTH_BOUND: record it as at the math/memory floor unless broader pointwise math-codegen or launch-overhead work moves the family."""
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


ROWS = 1024
COLS = 1536
INPUT_SHAPE = (ROWS, COLS)
VIEW_SHAPE = (32, 32, COLS)
OUTPUT_SHAPE = INPUT_SHAPE
OUTPUT_STRIDE = (COLS, 1)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _swiglu_pointwise_kernel(
        x_ptr,
        gate_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        swish = x * tl.sigmoid(x)
        tl.store(output_ptr + offsets, swish * gate, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_swiglu_pointwise.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mm_53, mm_54, shape0, shape1, shape2 = inputs
    for name, tensor in (("mm_53", mm_53), ("mm_54", mm_54)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be f32, got {tensor.dtype}")
        if tuple(tensor.shape) != INPUT_SHAPE:
            raise ValueError(f"{name} shape must be {INPUT_SHAPE}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")

    if tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape: {shape0}")
    if tuple(shape1) != VIEW_SHAPE:
        raise ValueError(f"unexpected second view shape: {shape1}")
    if tuple(shape2) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {shape2}")

    return mm_53, mm_54


@oracle_impl(hardware="H100", shapes="(T([1024, 1536], f32), T([1024, 1536], f32), S([32, 32, 1536]), S([32, 32, 1536]), S([1024, 1536]))")
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
    mm_53, mm_54 = _validate_inputs(inputs)
    n_elements = ROWS * COLS
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm_53.device,
        dtype=mm_53.dtype,
    )
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_N"]),)
    _swiglu_pointwise_kernel[grid](mm_53, mm_54, output, N=n_elements)
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
