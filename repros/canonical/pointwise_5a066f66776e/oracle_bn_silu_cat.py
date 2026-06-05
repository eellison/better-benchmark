"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle evaluates the full batchnorm-affine + SiLU + residual `cat` scope by folding `(conv - mean) * rsqrt(var + 1e-5) * weight + bias` into per-channel `conv * scale + shift`, then writing both concatenated halves directly in the final dense layout, whereas Inductor emits a small broadcast-parameter kernel and a generic pointwise/cat kernel that keeps the unfactored affine arithmetic in the full tensor loop; Inductor cannot do this today because its pointwise fusion canonicalization does not algebraically fold broadcasted normalization parameters through the final cat layout kernel; the fix is ALGEBRAIC_ELIMINATION: canonicalize this broadcast affine chain into per-channel scale/shift before scheduling the fused cat writer."""
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

EPS = 1.0e-5

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

    @triton.jit
    def _fold_affine_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        scale_ptr,
        shift_ptr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < C
        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
        var = tl.load(var_ptr + offsets, mask=mask, other=1.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        scale = weight / tl.sqrt(var + 1.0e-5)
        shift = bias - mean * scale
        tl.store(scale_ptr + offsets, scale, mask=mask)
        tl.store(shift_ptr + offsets, shift, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
        ],
        key=["TOTAL", "CHANNELS_LAST"],
    )
    @triton.jit
    def _bn_silu_cat_kernel(
        convolution_ptr,
        residual_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        conv_s2: tl.constexpr,
        conv_s3: tl.constexpr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
        residual_s3: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        TOTAL: tl.constexpr,
        CHANNELS_LAST: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        out_c = tl.full((BLOCK_N,), 0, tl.int64)
        out_h = tl.full((BLOCK_N,), 0, tl.int64)
        out_w = tl.full((BLOCK_N,), 0, tl.int64)
        out_n = tl.full((BLOCK_N,), 0, tl.int64)

        if CHANNELS_LAST:
            out_c = offsets % (2 * C)
            tmp = offsets // (2 * C)
            out_w = tmp % W
            tmp = tmp // W
            out_h = tmp % H
            out_n = tmp // H
        else:
            out_w = offsets % W
            tmp = offsets // W
            out_h = tmp % H
            tmp = tmp // H
            out_c = tmp % (2 * C)
            out_n = tmp // (2 * C)

        is_residual = out_c < C
        source_c = tl.where(is_residual, out_c, out_c - C)
        source_offset = (
            out_n * conv_s0
            + source_c * conv_s1
            + out_h * conv_s2
            + out_w * conv_s3
        )
        residual_offset = (
            out_n * residual_s0
            + source_c * residual_s1
            + out_h * residual_s2
            + out_w * residual_s3
        )

        residual_value = tl.load(
            residual_ptr + residual_offset,
            mask=mask & is_residual,
            other=0.0,
        )
        conv_value = tl.load(
            convolution_ptr + source_offset,
            mask=mask & ~is_residual,
            other=0.0,
        )
        scale = tl.load(scale_ptr + source_c, mask=mask & ~is_residual, other=0.0)
        shift = tl.load(shift_ptr + source_c, mask=mask & ~is_residual, other=0.0)

        affine = conv_value * scale + shift
        silu = affine / (1.0 + tl.exp(-affine))
        value = tl.where(is_residual, residual_value, silu)
        tl.store(output_ptr + offsets, value, mask=mask)


def _is_channels_last_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous(memory_format=torch.channels_last)


def _output_stride(
    batch: int,
    channels: int,
    height: int,
    width: int,
    channels_last: bool,
) -> tuple[int, int, int, int]:
    out_channels = channels * 2
    if channels_last:
        return (out_channels * height * width, 1, out_channels * width, out_channels)
    return (out_channels * height * width, height * width, width, 1)


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
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, convolution, var, weight, bias, residual = inputs
    if not all(isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError("all oracle inputs must be tensors")
    if convolution.dtype is not torch.float32 or residual.dtype is not torch.float32:
        raise TypeError("oracle expects float32 activation tensors")
    if tuple(convolution.shape) != tuple(residual.shape):
        raise ValueError("convolution and residual tensors must have the same shape")
    if convolution.device != residual.device or convolution.device.type != "cuda":
        raise ValueError("oracle expects CUDA tensors on one device")

    batch, channels, height, width = convolution.shape
    expected_param_shape = (channels,)
    for tensor in (mean, var, weight, bias):
        if tuple(tensor.shape) != expected_param_shape or tensor.dtype is not torch.float32:
            raise ValueError("unexpected affine parameter shape or dtype")

    channels_last = _is_channels_last_tensor(convolution) and _is_channels_last_tensor(residual)
    out_shape = (batch, channels * 2, height, width)
    out_stride = _output_stride(batch, channels, height, width, channels_last)

    scale = torch.empty((channels,), device=convolution.device, dtype=torch.float32)
    shift = torch.empty((channels,), device=convolution.device, dtype=torch.float32)
    output = torch.empty_strided(
        out_shape,
        out_stride,
        device=convolution.device,
        dtype=torch.float32,
    )

    block_c = triton.next_power_of_2(channels)
    _fold_affine_kernel[(1,)](
        mean,
        var,
        weight,
        bias,
        scale,
        shift,
        C=channels,
        BLOCK_C=block_c,
        num_warps=1,
        num_stages=1,
    )

    total = batch * channels * 2 * height * width
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    _bn_silu_cat_kernel[grid](
        convolution,
        residual,
        scale,
        shift,
        output,
        conv_s0=convolution.stride(0),
        conv_s1=convolution.stride(1),
        conv_s2=convolution.stride(2),
        conv_s3=convolution.stride(3),
        residual_s0=residual.stride(0),
        residual_s1=residual.stride(1),
        residual_s2=residual.stride(2),
        residual_s3=residual.stride(3),
        C=channels,
        H=height,
        W=width,
        TOTAL=total,
        CHANNELS_LAST=channels_last,
    )
    return output


def _check_output_values_and_layout(
    instance: torch.nn.Module,
    inputs,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        output = oracle_forward(inputs)
        torch.cuda.synchronize()

    _, convolution, _, _, _, residual = inputs
    batch, channels, height, width = convolution.shape
    channels_last = _is_channels_last_tensor(convolution) and _is_channels_last_tensor(residual)
    expected_shape = (batch, channels * 2, height, width)
    expected_stride = _output_stride(batch, channels, height, width, channels_last)
    layout_ok = tuple(output.shape) == expected_shape and output.stride() == expected_stride
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(output.shape)} stride={output.stride()})"
    )

    finite_mask = torch.isfinite(eager) & torch.isfinite(output)
    if finite_mask.any():
        max_diff = (eager[finite_mask].float() - output[finite_mask].float()).abs().max().item()
    else:
        max_diff = 0.0
    values_ok = torch.allclose(eager, output, atol=atol, rtol=rtol, equal_nan=True)
    print(f"  output 0 equal_nan: {'PASS' if values_ok else 'FAIL'} (max_finite_diff={max_diff:.2e})")
    return layout_ok and values_ok


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
        ok = _check_output_values_and_layout(instance, inputs, args.atol, args.rtol) and ok
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
