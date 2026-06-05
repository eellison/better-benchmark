"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT-J QKV layout transform by recognizing that the strided f32[1,16,128,256] input, permute [0,2,1,3], contiguous clone, two views, and final [4096,128] permute produce a fresh output whose physical storage order is exactly a linear copy of the input storage, whereas the best Inductor configuration already matches this full-scope storage-linear materialization on the measured shape; Inductor cannot get a confirmed local win here because the remaining cost is a required dense copy plus output metadata, and the oracle ties compile rather than establishing a faster floor; the fix is BANDWIDTH_BOUND: record this layout-materialization case as at floor and revisit only if broader copy/layout codegen changes beat the current compiled kernel on the exact full scope."""
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

if triton is not None:

    @triton.jit
    def _storage_linear_copy_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, values, mask=mask)


EXPECTED_INPUT_SHAPE = (1, 16, 128, 256)
EXPECTED_INPUT_STRIDE = (524288, 256, 4096, 1)
EXPECTED_SHAPE_0 = (1, 128, 4096)
EXPECTED_SHAPE_1 = (128, 4096)
OUTPUT_SHAPE = (4096, 128)
OUTPUT_STRIDE = (1, 4096)
NUMEL = 1 * 16 * 128 * 256
BLOCK_N = 1024


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    getitem_110, shape0, shape1 = inputs
    if not isinstance(getitem_110, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(getitem_110)!r}")
    if tuple(getitem_110.shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(getitem_110.shape)}")
    if tuple(getitem_110.stride()) != EXPECTED_INPUT_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(getitem_110.stride())}")
    if getitem_110.storage_offset() != 0:
        raise ValueError(f"unexpected input storage_offset: {getitem_110.storage_offset()}")
    if getitem_110.dtype is not torch.float32:
        raise TypeError(f"expected torch.float32 input, got {getitem_110.dtype}")
    if not getitem_110.is_cuda:
        raise ValueError("oracle_qkv_layout_transpose.py expects CUDA inputs")
    if _shape_tuple(shape0, "_shape_param_0") != EXPECTED_SHAPE_0:
        raise ValueError(f"unexpected _shape_param_0: {_shape_tuple(shape0, '_shape_param_0')}")
    if _shape_tuple(shape1, "_shape_param_1") != EXPECTED_SHAPE_1:
        raise ValueError(f"unexpected _shape_param_1: {_shape_tuple(shape1, '_shape_param_1')}")

    return getitem_110


def oracle_forward(inputs):
    """Run the full Repro.forward scope as a storage-linear materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    getitem_110 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=getitem_110.device,
        dtype=getitem_110.dtype,
    )
    grid = (triton.cdiv(NUMEL, BLOCK_N),)
    _storage_linear_copy_kernel[grid](
        getitem_110,
        output,
        N=NUMEL,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
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
