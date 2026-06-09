"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete per-channel BN-inference affine, SiLU-style divide by exp denominator, and right/bottom constant-pad store in one layout-aware Triton kernel, whereas Inductor currently lowers this repro as a producer pointwise kernel for the affine tensor followed by a separate consumer kernel for exp/div plus the padded output layout; Inductor cannot do this today because the pointwise scheduler does not fuse through constant_pad_nd's output-indexing stencil when the producer and consumer iteration domains have different shapes/layouts; the fix is SCHEDULER_FUSION: teach scheduler/codegen to recognize zero-pad layout stencils and generate one fused kernel that maps padded output indices back to the valid producer region."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=["C", "H", "W"],
    )
    @triton.jit
    def _bn_silu_pad_nchw_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        H_OUT: tl.constexpr,
        W_OUT: tl.constexpr,
        X_S0: tl.constexpr,
        X_S1: tl.constexpr,
        X_S2: tl.constexpr,
        X_S3: tl.constexpr,
        O_S0: tl.constexpr,
        O_S1: tl.constexpr,
        O_S2: tl.constexpr,
        O_S3: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        h_offsets = hw_offsets // W_OUT
        w_offsets = hw_offsets - h_offsets * W_OUT

        c_mask = c_offsets < C
        hw_mask = hw_offsets < (H_OUT * W_OUT)
        out_mask = c_mask[:, None] & hw_mask[None, :]
        data_mask = out_mask & (h_offsets[None, :] < H) & (w_offsets[None, :] < W)

        x_offsets = (
            n * X_S0
            + c_offsets[:, None] * X_S1
            + h_offsets[None, :] * X_S2
            + w_offsets[None, :] * X_S3
        )
        out_offsets = (
            n * O_S0
            + c_offsets[:, None] * O_S1
            + h_offsets[None, :] * O_S2
            + w_offsets[None, :] * O_S3
        )

        x = tl.load(x_ptr + x_offsets, mask=data_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        affine = (x - mean[:, None]) * (1.0 / tl.sqrt(var[:, None] + 0.001))
        affine = affine * weight[:, None] + bias[:, None]
        out = affine / (tl.exp2(-affine * 1.4426950408889634) + 1.0)
        out = tl.where(data_mask, out, 0.0)
        tl.store(out_ptr + out_offsets, out, mask=out_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_HW": 16, "BLOCK_C": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_HW": 32, "BLOCK_C": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_HW": 32, "BLOCK_C": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_HW": 64, "BLOCK_C": 32}, num_warps=8, num_stages=3),
        ],
        key=["C", "H", "W"],
    )
    @triton.jit
    def _bn_silu_pad_nhwc_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        H_OUT: tl.constexpr,
        W_OUT: tl.constexpr,
        X_S0: tl.constexpr,
        X_S1: tl.constexpr,
        X_S2: tl.constexpr,
        X_S3: tl.constexpr,
        O_S0: tl.constexpr,
        O_S1: tl.constexpr,
        O_S2: tl.constexpr,
        O_S3: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        n = tl.program_id(0)
        hw_block = tl.program_id(1)
        c_block = tl.program_id(2)

        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        h_offsets = hw_offsets // W_OUT
        w_offsets = hw_offsets - h_offsets * W_OUT

        hw_mask = hw_offsets < (H_OUT * W_OUT)
        c_mask = c_offsets < C
        out_mask = hw_mask[:, None] & c_mask[None, :]
        data_mask = out_mask & (h_offsets[:, None] < H) & (w_offsets[:, None] < W)

        x_offsets = (
            n * X_S0
            + c_offsets[None, :] * X_S1
            + h_offsets[:, None] * X_S2
            + w_offsets[:, None] * X_S3
        )
        out_offsets = (
            n * O_S0
            + c_offsets[None, :] * O_S1
            + h_offsets[:, None] * O_S2
            + w_offsets[:, None] * O_S3
        )

        x = tl.load(x_ptr + x_offsets, mask=data_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        affine = (x - mean[None, :]) * (1.0 / tl.sqrt(var[None, :] + 0.001))
        affine = affine * weight[None, :] + bias[None, :]
        out = affine / (tl.exp2(-affine * 1.4426950408889634) + 1.0)
        out = tl.where(data_mask, out, 0.0)
        tl.store(out_ptr + out_offsets, out, mask=out_mask)


def _require_channel_vector(
    name: str,
    value: Any,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (channels,):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {(channels,)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_fused_bn_silu_pad.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_25 must be a tensor, got {type(x)!r}")
    if x.dim() != 4:
        raise ValueError(f"convolution_25 must be rank 4, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"convolution_25 has dtype {x.dtype}, expected torch.float32")
    if not x.is_cuda:
        raise RuntimeError("Triton oracle requires CUDA inputs")

    _, channels, _, _ = x.shape
    mean_t = _require_channel_vector("arg97_1", mean, channels, x.device)
    var_t = _require_channel_vector("arg98_1", var, channels, x.device)
    weight_t = _require_channel_vector("arg99_1", weight, channels, x.device)
    bias_t = _require_channel_vector("arg100_1", bias, channels, x.device)

    if x.is_contiguous(memory_format=torch.channels_last):
        layout = 1
    elif x.is_contiguous():
        layout = 0
    else:
        raise ValueError(f"unsupported convolution_25 stride: {tuple(x.stride())}")

    return mean_t, x, var_t, weight_t, bias_t, layout


def _padded_output(
    x: torch.Tensor,
    layout: int,
) -> torch.Tensor:
    batch, channels, height, width = x.shape
    out_shape = (batch, channels, height + 1, width + 1)
    if layout == 1:
        out_stride = ((height + 1) * (width + 1) * channels, 1, (width + 1) * channels, channels)
    else:
        out_stride = (channels * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1)
    return torch.empty_strided(out_shape, out_stride, device=x.device, dtype=x.dtype)


@oracle_impl(hardware="H100", shapes="(T([240], f32), T([128, 240, 28, 28], f32), T([240], f32), T([240], f32), T([240], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward BN-inference, SiLU, and constant-pad scope."""
    mean, x, var, weight, bias, layout = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    out = _padded_output(x, layout)
    height_out = height + 1
    width_out = width + 1

    if layout == 1:
        grid = lambda meta: (
            batch,
            triton.cdiv(height_out * width_out, meta["BLOCK_HW"]),
            triton.cdiv(channels, meta["BLOCK_C"]),
        )
        _bn_silu_pad_nhwc_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            out,
            C=channels,
            H=height,
            W=width,
            H_OUT=height_out,
            W_OUT=width_out,
            X_S0=x.stride(0),
            X_S1=x.stride(1),
            X_S2=x.stride(2),
            X_S3=x.stride(3),
            O_S0=out.stride(0),
            O_S1=out.stride(1),
            O_S2=out.stride(2),
            O_S3=out.stride(3),
        )
    else:
        grid = lambda meta: (
            batch,
            triton.cdiv(channels, meta["BLOCK_C"]),
            triton.cdiv(height_out * width_out, meta["BLOCK_HW"]),
        )
        _bn_silu_pad_nchw_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            out,
            C=channels,
            H=height,
            W=width,
            H_OUT=height_out,
            W_OUT=width_out,
            X_S0=x.stride(0),
            X_S1=x.stride(1),
            X_S2=x.stride(2),
            X_S3=x.stride(3),
            O_S0=out.stride(0),
            O_S1=out.stride(1),
            O_S2=out.stride(2),
            O_S3=out.stride(3),
        )
    return out


def _check_oracle(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_ok = torch.equal(expected_nan, actual_nan)
    compare_mask = ~expected_nan & ~actual_nan
    finite_mask = torch.isfinite(expected_f32) & torch.isfinite(actual_f32)

    if compare_mask.any():
        values_ok = torch.allclose(expected_f32[compare_mask], actual_f32[compare_mask], atol=atol, rtol=rtol)
    else:
        values_ok = True
    if finite_mask.any():
        max_diff = (expected_f32[finite_mask] - actual_f32[finite_mask]).abs().max().item()
    else:
        max_diff = 0.0

    layout_ok = shape_ok and dtype_ok and stride_ok
    print(
        f"  output 0 values: {'PASS' if values_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} dtype={actual.dtype} max_finite_diff={max_diff:.2e})"
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    print(
        f"  output 0 NaNs: {'PASS' if nan_ok else 'FAIL'} "
        f"(expected_nan={int(expected_nan.sum().item())}, oracle_nan={int(actual_nan.sum().item())})"
    )
    return values_ok and layout_ok and nan_ok


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
        ok = _check_oracle(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
