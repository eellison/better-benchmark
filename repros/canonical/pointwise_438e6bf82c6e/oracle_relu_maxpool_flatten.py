"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ReLU -> 2x2 stride-2 low-memory maxpool-with-offsets -> shape-identical adaptive_avg_pool2d -> view scope in one Triton kernel, writing the int8 pool offsets and flattened pooled tensor directly, whereas Inductor currently treats the maxpool stencil and following layout/materialization work as separate generic scheduling regions; Inductor cannot do this today because the scheduler does not fuse pointwise producers and identity/layout consumers through a multi-output low-memory maxpool while preserving first-index tie and NaN offset semantics; the fix is SCHEDULER_FUSION: allow the maxpool lowering to absorb the ReLU producer and emit both offsets and the final flattened layout from the same loop nest."""
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


BATCH = 128
CHANNELS = 512
H_IN = 14
W_IN = 14
H_OUT = 7
W_OUT = 7
INPUT_SHAPE = (BATCH, CHANNELS, H_IN, W_IN)
INPUT_STRIDE = (CHANNELS * H_IN * W_IN, H_IN * W_IN, W_IN, 1)
OFFSET_SHAPE = (BATCH, CHANNELS, H_OUT, W_OUT)
OFFSET_STRIDE = (CHANNELS * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1)
FLAT_SHAPE = (BATCH, CHANNELS * H_OUT * W_OUT)
FLAT_STRIDE = (CHANNELS * H_OUT * W_OUT, 1)
TOTAL_OUTPUTS = BATCH * CHANNELS * H_OUT * W_OUT
BLOCK_N = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _take_candidate(candidate, candidate_offset, best, best_offset):
        take = (candidate > best) | ((best == best) & (candidate != candidate))
        best = tl.where(take, candidate, best)
        best_offset = tl.where(take, candidate_offset, best_offset)
        return best, best_offset


    @triton.jit
    def _relu_maxpool_flatten_kernel(
        input_ptr,
        offsets_ptr,
        flat_values_ptr,
        BLOCK_N_: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)

        out_w = linear % 7
        tmp = linear // 7
        out_h = tmp % 7
        plane = tmp // 7

        input_base = plane * 196 + out_h * 28 + out_w * 2
        x00 = tl.load(input_ptr + input_base).to(tl.float32)
        x01 = tl.load(input_ptr + input_base + 1).to(tl.float32)
        x10 = tl.load(input_ptr + input_base + 14).to(tl.float32)
        x11 = tl.load(input_ptr + input_base + 15).to(tl.float32)

        best = _relu_preserve_nan(x00)
        best_offset = tl.zeros((BLOCK_N_,), dtype=tl.int32)

        candidate = _relu_preserve_nan(x01)
        best, best_offset = _take_candidate(candidate, 1, best, best_offset)
        candidate = _relu_preserve_nan(x10)
        best, best_offset = _take_candidate(candidate, 2, best, best_offset)
        candidate = _relu_preserve_nan(x11)
        best, best_offset = _take_candidate(candidate, 3, best, best_offset)

        tl.store(offsets_ptr + linear, best_offset.to(tl.int8))
        tl.store(flat_values_ptr + linear, best)


def _torch_oracle(convolution_12, shape_param):
    relu = torch.relu(convolution_12)
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu, [2, 2], [2, 2], [0, 0], [1, 1], False
    )
    pooled = torch.ops.aten._adaptive_avg_pool2d.default(values, [7, 7])
    return offsets, torch.ops.aten.view.default(pooled, shape_param)


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    convolution_12, shape_param = inputs
    if not isinstance(convolution_12, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(convolution_12)!r}")
    if tuple(convolution_12.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(convolution_12.shape)}")
    if tuple(convolution_12.stride()) != INPUT_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(convolution_12.stride())}")
    if convolution_12.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected float16 or float32 input, got {convolution_12.dtype}")
    if tuple(shape_param) != FLAT_SHAPE:
        raise ValueError(f"unexpected view shape: {shape_param}")
    return convolution_12, shape_param


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
    convolution_12, shape_param = _validate_inputs(inputs)
    if not convolution_12.is_cuda:
        return _torch_oracle(convolution_12, shape_param)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    offsets = torch.empty_strided(
        OFFSET_SHAPE,
        OFFSET_STRIDE,
        device=convolution_12.device,
        dtype=torch.int8,
    )
    flat_values = torch.empty_strided(
        FLAT_SHAPE,
        FLAT_STRIDE,
        device=convolution_12.device,
        dtype=convolution_12.dtype,
    )

    _relu_maxpool_flatten_kernel[(TOTAL_OUTPUTS // BLOCK_N,)](
        convolution_12,
        offsets,
        flat_values,
        BLOCK_N_=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return offsets, flat_values


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
