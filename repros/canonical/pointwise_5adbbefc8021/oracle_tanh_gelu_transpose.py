"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DistilGPT2 tanh-approximate GELU scope in one storage-linear Triton pointwise kernel, including the metadata-only `[16384,3072] -> [32,512,3072] -> [16384,3072]` views and the returned `[3072,16384]` transpose layout with stride `(1,3072)`, whereas Inductor already lowers the same isolated view-plus-pointwise-plus-permute graph as one fused pointwise kernel over the required f32 read, tanh-heavy math, and f32 output storage; Inductor cannot materially do less local work today because there is no producer, consumer, reduction, scatter, or removable intermediate materialization left inside this repro beyond the required math and memory traffic; the fix is BANDWIDTH_BOUND: record this as a full-scope pointwise/layout floor unless broader pointwise transcendental codegen or launch-overhead improvements move both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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

INPUT_SHAPE = (16384, 3072)
VIEW_SHAPE = (32, 512, 3072)
OUTPUT_SHAPE = (3072, 16384)
INPUT_STRIDE = (3072, 1)
OUTPUT_STRIDE = (1, 3072)
N_ELEMENTS = 16384 * 3072

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _tanh_gelu_transpose_kernel(
        input_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x3 = x * x * x
        tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
        y = (x * 0.5) * (libdevice.tanh(tanh_arg) + 1.0)

        tl.store(output_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(x).__name__}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} input shape {tuple(x.shape)} != {INPUT_SHAPE}")
    if tuple(x.stride()) != INPUT_STRIDE:
        raise ValueError(f"{REPRO_ID} input stride {tuple(x.stride())} != {INPUT_STRIDE}")
    if x.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} input dtype {x.dtype} != torch.float32")
    if not x.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA input")

    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"{REPRO_ID} got unexpected first view shape: {shape0!r}")
    if _shape_tuple(shape1) != (-1, INPUT_SHAPE[1]):
        raise ValueError(f"{REPRO_ID} got unexpected output view shape: {shape1!r}")
    return x


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
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
        raise RuntimeError("Triton is required for oracle_tanh_gelu_transpose.py")

    x = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_SIZE"]),)
    _tanh_gelu_transpose_kernel[grid](
        x,
        output,
        n_elements=N_ELEMENTS,
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
