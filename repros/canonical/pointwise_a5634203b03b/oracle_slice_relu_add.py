"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs slice/ReLU/add pointwise scope with one Triton kernel that tiles by `[batch*channel, time]`, loads `arg26_1[:, :, 1426:-1426]` directly, applies NaN-preserving ReLU to `convolution_9`, and materializes the exact contiguous output, whereas Inductor already lowers the same full expression to an equally efficient fused pointwise kernel; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new semantic pattern because the mandatory two input reads plus one output write dominate CUDAGraph timing; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise bandwidth/codegen improvements move both implementations."""
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
CHANNELS = 64
TIME = 92844
ARG_TIME = 95696
SLICE_START = 1426
OUTPUT_SHAPE_SUFFIX = (CHANNELS, TIME)
OUTPUT_STRIDE_SUFFIX = (TIME, 1)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_T": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_T": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_T": 4096}, num_warps=8, num_stages=4),
        ],
        key=["T"],
    )
    @triton.jit
    def _slice_relu_add_kernel(
        conv_ptr,
        arg_ptr,
        out_ptr,
        T: tl.constexpr,
        ARG_T: tl.constexpr,
        SLICE_START_: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = cols < T

        conv_offsets = row * T + cols
        arg_offsets = row * ARG_T + SLICE_START_ + cols

        conv = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0).to(tl.float32)
        sliced = tl.load(arg_ptr + arg_offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.where(conv != conv, conv, tl.maximum(conv, 0.0))
        tl.store(out_ptr + conv_offsets, relu + sliced, mask=mask)


def _require_tensor(
    name: str,
    value,
    expected_shape: tuple[int, ...],
    expected_stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be a CUDA tensor, got {value.device}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {tuple(value.shape)}")
    if tuple(value.stride()) != expected_stride:
        raise ValueError(f"{name} stride must be {expected_stride}, got {tuple(value.stride())}")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} storage_offset must be 0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs) -> tuple[torch.Tensor, torch.Tensor, int]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    convolution_9, arg26_1 = inputs
    if not isinstance(convolution_9, torch.Tensor):
        raise TypeError(f"convolution_9 must be a torch.Tensor, got {type(convolution_9)!r}")
    if convolution_9.ndim != 3:
        raise ValueError(f"convolution_9 must be rank 3, got shape={tuple(convolution_9.shape)}")
    batch = int(convolution_9.shape[0])
    conv_shape = (batch, CHANNELS, TIME)
    arg_shape = (batch, CHANNELS, ARG_TIME)
    conv_stride = (CHANNELS * TIME, TIME, 1)
    arg_stride = (CHANNELS * ARG_TIME, ARG_TIME, 1)

    conv = _require_tensor("convolution_9", convolution_9, conv_shape, conv_stride)
    arg = _require_tensor("arg26_1", arg26_1, arg_shape, arg_stride)
    if arg.device != conv.device:
        raise ValueError(f"input devices must match, got {conv.device} and {arg.device}")
    return conv, arg, batch


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
    conv, arg, batch = _validate_inputs(inputs)
    output = torch.empty_strided(
        (batch, *OUTPUT_SHAPE_SUFFIX),
        (CHANNELS * TIME, *OUTPUT_STRIDE_SUFFIX),
        device=conv.device,
        dtype=conv.dtype,
    )

    rows = batch * CHANNELS
    grid = lambda meta: (rows, triton.cdiv(TIME, meta["BLOCK_T"]))
    _slice_relu_add_kernel[grid](
        conv,
        arg,
        output,
        T=TIME,
        ARG_T=ARG_TIME,
        SLICE_START_=SLICE_START,
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
