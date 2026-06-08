"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete right-and-bottom zero `constant_pad_nd` scope by directly materializing PyTorch's contiguous-or-channels-last padded output in one storage-order Triton kernel, whereas Inductor already lowers this isolated pad to an equivalent one-kernel output materialization with essentially the same CUDAGraph timing; Inductor cannot materially improve this local repro today because the exact contract still requires a fresh padded allocation, reading every interior input element, writing every output element, and zeroing the narrow pad fringe with no producer fusion or algebraic work left to remove; the fix is BANDWIDTH_BOUND: record this as an at-floor pad-copy case unless broader pointwise memory-codegen, allocation, or launch-overhead work moves both implementations together."""
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


BLOCK_SIZE = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _pad_nchw_kernel(
        input_ptr,
        output_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        out_h: tl.constexpr = height + 1
        out_w: tl.constexpr = width + 1

        w = offsets % out_w
        tmp = offsets // out_w
        h = tmp % out_h
        tmp = tmp // out_h
        c = tmp % channels
        n = tmp // channels

        in_bounds = mask & (h < height) & (w < width)
        in_offsets = ((n * channels + c) * height + h) * width + w
        values = tl.load(input_ptr + in_offsets, mask=in_bounds, other=0.0)
        values = tl.where(in_bounds, values, 0.0)
        tl.store(output_ptr + offsets, values, mask=mask)

    @triton.jit
    def _pad_nhwc_kernel(
        input_ptr,
        output_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total

        out_h: tl.constexpr = height + 1
        out_w: tl.constexpr = width + 1

        c = offsets % channels
        tmp = offsets // channels
        w = tmp % out_w
        tmp = tmp // out_w
        h = tmp % out_h
        n = tmp // out_h

        in_bounds = mask & (h < height) & (w < width)
        in_offsets = ((n * height + h) * width + w) * channels + c
        values = tl.load(input_ptr + in_offsets, mask=in_bounds, other=0.0)
        values = tl.where(in_bounds, values, 0.0)
        tl.store(output_ptr + offsets, values, mask=mask)


def _output_layout(x: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    batch, channels, height, width = x.shape
    out_h = height + 1
    out_w = width + 1
    shape = (batch, channels, out_h, out_w)

    if x.is_contiguous(memory_format=torch.channels_last):
        stride = (channels * out_h * out_w, 1, out_w * channels, channels)
        return shape, stride, True
    if x.is_contiguous():
        stride = (channels * out_h * out_w, out_h * out_w, out_w, 1)
        return shape, stride, False
    raise ValueError(
        f"{REPRO_ID} expects contiguous NCHW or channels-last input, "
        f"got stride {tuple(x.stride())}"
    )


def _expect_input(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(value)!r}")
    if value.dim() != 4:
        raise ValueError(f"{REPRO_ID} expects a 4D NCHW input, got shape {tuple(value.shape)}")
    if value.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"{REPRO_ID} expects f16 or f32 input, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    return value


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
        raise RuntimeError("Triton is required for oracle_right_bottom_pad_copy.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = _expect_input(inputs[0])
    out_shape, out_stride, channels_last = _output_layout(x)
    output = torch.empty_strided(out_shape, out_stride, device=x.device, dtype=x.dtype)

    _, channels, height, width = x.shape
    total = output.numel()

    grid = (triton.cdiv(total, BLOCK_SIZE),)
    if channels_last:
        _pad_nhwc_kernel[grid](
            x,
            output,
            total=total,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _pad_nchw_kernel[grid](
            x,
            output,
            total=total,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
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
