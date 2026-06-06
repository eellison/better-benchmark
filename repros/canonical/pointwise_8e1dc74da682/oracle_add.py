"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete isolated two-input pointwise add from Repro.forward as one storage-linear Triton kernel, preserving the fresh dense output layout chosen by eager aten.add for both contiguous and channels-last dense inputs, whereas Inductor already lowers this one-op repro to the same single pointwise-kernel and mandatory two-read/one-write memory-traffic envelope; Inductor cannot materially improve this local case through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new pattern because there is no surrounding producer or consumer to fuse and no redundant arithmetic to remove; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise memory-bandwidth case unless broader pointwise codegen or launch-overhead improvements move both implementations."""
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
BLOCK_SIZE = 1024


if triton is not None:

    @triton.jit
    def _add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def _storage_span(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return sum((dim - 1) * st for dim, st in zip(shape, stride) if dim > 1) + 1


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_add.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, y = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects two tensor inputs")
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if x.device != y.device:
        raise ValueError(f"input devices differ: {x.device} vs {y.device}")
    if x.dtype != y.dtype:
        raise TypeError(f"input dtypes differ: {x.dtype} vs {y.dtype}")
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError(f"input shapes differ: {tuple(x.shape)} vs {tuple(y.shape)}")
    if tuple(x.stride()) != tuple(y.stride()):
        raise ValueError(f"input strides differ: {tuple(x.stride())} vs {tuple(y.stride())}")
    if x.storage_offset() != 0 or y.storage_offset() != 0:
        raise ValueError("oracle_add.py expects zero-storage-offset dense inputs")

    shape = tuple(int(dim) for dim in x.shape)
    stride = tuple(int(st) for st in x.stride())
    if _storage_span(shape, stride) != x.numel():
        raise ValueError(f"input stride {stride} for shape {shape} is not dense")

    return x, y, shape, stride


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
    x, y, shape, stride = _validate_inputs(inputs)
    output = torch.empty_strided(
        shape,
        stride,
        device=x.device,
        dtype=x.dtype,
    )
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    _add_kernel[grid](
        x,
        y,
        output,
        n_elements=x.numel(),
        BLOCK_N=BLOCK_SIZE,
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
