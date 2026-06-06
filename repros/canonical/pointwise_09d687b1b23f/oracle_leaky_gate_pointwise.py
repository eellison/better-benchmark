"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured leaky gate `where(arg9_1 > 0, getitem_9, getitem_9 * 0.2)` in one storage-linear Triton pointwise kernel over the required contiguous f32 `[1024,64,32,32]` output, whereas Inductor already lowers this isolated pointwise graph to the same essential fused one-kernel, two-input-read, one-output-write schedule; Inductor cannot materially improve this specific repro with a scheduler-fusion, scatter, split-K, algebraic-elimination, recompute, or new-pattern change because the full scope is dominated by mandatory global memory traffic and launch overhead rather than avoidable intermediates; the fix is BANDWIDTH_BOUND: treat this as an at-floor pointwise case unless broader pointwise codegen or memory-throughput improvements move both implementations."""
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


INPUT_SHAPE = (1024, 64, 32, 32)
OUTPUT_SHAPE = INPUT_SHAPE
CONTIGUOUS_STRIDE = (64 * 32 * 32, 32 * 32, 32, 1)
N_ELEMENTS = 1024 * 64 * 32 * 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 8192}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def oracle_kernel(
        predicate_ptr,
        value_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        predicate_value = tl.load(predicate_ptr + offsets, mask=mask, other=0.0)
        value = tl.load(value_ptr + offsets, mask=mask, other=0.0)
        output = tl.where(predicate_value > 0.0, value, value * 0.2)
        tl.store(output_ptr + offsets, output, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_leaky_gate_pointwise.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg9_1, getitem_9 = inputs
    if not isinstance(arg9_1, torch.Tensor):
        raise TypeError(f"arg9_1 must be a tensor, got {type(arg9_1)!r}")
    if not isinstance(getitem_9, torch.Tensor):
        raise TypeError(f"getitem_9 must be a tensor, got {type(getitem_9)!r}")
    for name, tensor in (("arg9_1", arg9_1), ("getitem_9", getitem_9)):
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be f32, got {tensor.dtype}")
        if tuple(tensor.shape) != INPUT_SHAPE:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {INPUT_SHAPE}")
        if tuple(tensor.stride()) != CONTIGUOUS_STRIDE:
            raise ValueError(
                f"{name} has stride {tuple(tensor.stride())}, expected {CONTIGUOUS_STRIDE}"
            )
    return arg9_1, getitem_9


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
    arg9_1, getitem_9 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        CONTIGUOUS_STRIDE,
        device=getitem_9.device,
        dtype=getitem_9.dtype,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    oracle_kernel[grid](arg9_1, getitem_9, output, N=N_ELEMENTS)
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
