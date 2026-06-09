"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BN-inference affine plus ReLU pointwise scope with one layout-aware Triton materialization that tiles by channel and spatial position, reuses each channel's mean/variance/weight/bias across the tile, and preserves either contiguous NCHW or dense channels-last output strides, whereas Inductor's generic fused pointwise schedule already reaches the same practical one-launch floor for the full tensor and layout materialization; Inductor cannot materially improve this local repro today because the remaining work is dominated by mandatory f32 input/output traffic, channel-parameter reads, sqrt latency, and launch overhead rather than an avoidable intermediate; the fix is BANDWIDTH_BOUND: record this as at floor unless a broader launch-overhead or memory-bandwidth change moves both paths."""
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=["C", "HW"],
    )
    @triton.jit
    def _bn_affine_relu_nchw_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        c_mask = c_offsets < C
        hw_mask = hw_offsets < HW
        mask = c_mask[:, None] & hw_mask[None, :]

        offsets = n * C * HW + c_offsets[:, None] * HW + hw_offsets[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + 1.0e-3)
        y = (x - mean[:, None]) * invstd[:, None] * weight[:, None] + bias[:, None]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        tl.store(out_ptr + offsets, y, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 64, "BLOCK_HW": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 64, "BLOCK_HW": 32}, num_warps=8, num_stages=3),
        ],
        key=["C", "HW"],
    )
    @triton.jit
    def _bn_affine_relu_channels_last_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        hw_block = tl.program_id(1)
        c_block = tl.program_id(2)

        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_mask = hw_offsets < HW
        c_mask = c_offsets < C
        mask = hw_mask[:, None] & c_mask[None, :]

        offsets = n * C * HW + hw_offsets[:, None] * C + c_offsets[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + 1.0e-3)
        y = (x - mean[None, :]) * invstd[None, :] * weight[None, :] + bias[None, :]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        tl.store(out_ptr + offsets, y, mask=mask)


def _dense_nchw_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (channels * height * width, height * width, width, 1)


def _dense_channels_last_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _, channels, height, width = shape
    return (height * width * channels, 1, width * channels, channels)


def _require_f32_channel_vector(
    name: str,
    value: Any,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (channels,):
        raise ValueError(f"{name} shape must be ({channels},), got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_affine_relu_layout.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_90 must be a tensor, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("oracle_bn_affine_relu_layout.py expects CUDA inputs")
    if x.dtype != torch.float32:
        raise TypeError(f"convolution_90 must be torch.float32, got {x.dtype}")
    if x.dim() != 4:
        raise ValueError(f"convolution_90 must be rank 4, got shape {tuple(x.shape)}")

    shape = tuple(x.shape)
    batch, channels, height, width = shape
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"convolution_90 has invalid shape {shape}")

    stride = tuple(x.stride())
    nchw_stride = _dense_nchw_stride(shape)
    channels_last_stride = _dense_channels_last_stride(shape)
    if stride == nchw_stride:
        is_channels_last = False
    elif stride == channels_last_stride:
        is_channels_last = True
    else:
        raise ValueError(
            f"unsupported convolution_90 stride {stride}; expected dense NCHW "
            f"{nchw_stride} or dense channels-last {channels_last_stride}"
        )

    mean_t = _require_f32_channel_vector("arg452_1", mean, channels, x.device)
    var_t = _require_f32_channel_vector("arg453_1", var, channels, x.device)
    weight_t = _require_f32_channel_vector("arg454_1", weight, channels, x.device)
    bias_t = _require_f32_channel_vector("arg455_1", bias, channels, x.device)
    return mean_t, x, var_t, weight_t, bias_t, is_channels_last


@oracle_impl(hardware="H100", shapes="(T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32))")
def oracle_forward(inputs):
    """Run the complete Repro.forward BN affine plus ReLU scope."""
    mean, x, var, weight, bias, is_channels_last = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    hw = height * width

    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    if is_channels_last:
        grid = lambda meta: (
            batch,
            triton.cdiv(hw, meta["BLOCK_HW"]),
            triton.cdiv(channels, meta["BLOCK_C"]),
        )
        _bn_affine_relu_channels_last_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            output,
            C=channels,
            HW=hw,
        )
    else:
        grid = lambda meta: (
            batch,
            triton.cdiv(channels, meta["BLOCK_C"]),
            triton.cdiv(hw, meta["BLOCK_HW"]),
        )
        _bn_affine_relu_nchw_kernel[grid](
            mean,
            x,
            var,
            weight,
            bias,
            output,
            C=channels,
            HW=hw,
        )
    return output


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _check_oracle_nan_aware(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_ok = True
    for index, (expected_item, actual_item) in enumerate(zip(expected_list, actual_list)):
        if not isinstance(expected_item, torch.Tensor) or not isinstance(actual_item, torch.Tensor):
            ok = expected_item == actual_item
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_ok = all_ok and bool(ok)
            continue

        metadata_ok = (
            tuple(expected_item.shape) == tuple(actual_item.shape)
            and expected_item.dtype == actual_item.dtype
            and expected_item.stride() == actual_item.stride()
            and expected_item.device == actual_item.device
        )
        if not metadata_ok:
            print(
                f"  output {index}: SCOPE_MISMATCH "
                f"oracle=(shape={list(actual_item.shape)} stride={actual_item.stride()} "
                f"dtype={actual_item.dtype}) "
                f"eager=(shape={list(expected_item.shape)} stride={expected_item.stride()} "
                f"dtype={expected_item.dtype})"
            )
            all_ok = False
            continue

        if not expected_item.is_floating_point():
            values_ok = torch.equal(expected_item, actual_item)
            print(f"  output {index}: {'PASS' if values_ok else 'FAIL'} (exact)")
            all_ok = all_ok and values_ok
            continue

        expected_f32 = expected_item.float()
        actual_f32 = actual_item.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_ok = torch.equal(expected_nan, actual_nan)
        finite = ~expected_nan & ~actual_nan
        if finite.any():
            finite_expected = expected_f32[finite]
            finite_actual = actual_f32[finite]
            max_diff = (finite_expected - finite_actual).abs().max().item()
            values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            values_ok = True

        print(
            f"  output {index} values: {'PASS' if values_ok else 'FAIL'} "
            f"(shape={list(actual_item.shape)} dtype={actual_item.dtype} "
            f"max_finite_diff={max_diff:.2e})"
        )
        print(
            f"  output {index} layout: PASS "
            f"(stride={actual_item.stride()})"
        )
        print(
            f"  output {index} NaNs: {'PASS' if nan_ok else 'FAIL'} "
            f"(expected_nan={int(expected_nan.sum().item())}, "
            f"oracle_nan={int(actual_nan.sum().item())})"
        )
        all_ok = all_ok and values_ok and nan_ok

    return all_ok


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
        ok = _check_oracle_nan_aware(
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
