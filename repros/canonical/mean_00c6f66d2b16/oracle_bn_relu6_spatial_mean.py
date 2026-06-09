"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileNetV2 BN-inference affine, ReLU6 clamp, 7x7 spatial mean, as_strided-compatible reshape, and final contiguous f32 [128,1280] output in one row-tiled Triton kernel that reuses per-channel BN parameters across each spatial reduction, whereas Inductor's generated norm-template lowering remains a slightly slower generic fused small-spatial reduction for the same channels-last input layout; Inductor cannot emit this exact floor today because the scheduler does not have a guarded BN-inference plus activation plus fixed spatial-mean template that folds the broadcast parameters into the row reduction and writes the final view layout directly; the fix is SCHEDULER_FUSION: add a benchmark-gated BN-affine ReLU6 spatial-mean schedule for NCHW tensors with channels-last or contiguous strides."""
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


# --- Oracle kernel(s) ---
BATCH = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CHANNELS_LAST_X_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
CONTIGUOUS_X_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
BLOCK_ROWS = 32
BLOCK_HW = 64

if triton is not None:

    @triton.jit
    def _bn_relu6_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        batch: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        h_offsets = hw_offsets // width
        w_offsets = hw_offsets - h_offsets * width

        n_offsets = row_offsets // channels
        channel_offsets = row_offsets - n_offsets * channels
        valid_rows = row_offsets < batch * channels
        valid_hw = hw_offsets < hw

        x_offsets = (
            n_offsets[:, None] * x_stride_n
            + channel_offsets[:, None] * x_stride_c
            + h_offsets[None, :] * x_stride_h
            + w_offsets[None, :] * x_stride_w
        )
        valid = valid_rows[:, None] & valid_hw[None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        affine = (x - mean[:, None]) * invstd[:, None]
        affine = affine * weight[:, None] + bias[:, None]
        relu6 = tl.where(affine < 0.0, 0.0, affine)
        relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
        reduced = tl.sum(tl.where(valid_hw[None, :], relu6, 0.0), axis=1) * (1.0 / 49.0)

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    allowed_strides: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if allowed_strides is not None and tuple(value.stride()) not in allowed_strides:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected one of {allowed_strides}"
        )
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, view_shape = inputs
    mean_t = _require_f32_tensor("arg257_1", mean, (CHANNELS,), ((1,),))
    x_t = _require_f32_tensor(
        "convolution_51",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS_LAST_X_STRIDE, CONTIGUOUS_X_STRIDE),
    )
    var_t = _require_f32_tensor("arg258_1", var, (CHANNELS,), ((1,),))
    weight_t = _require_f32_tensor("arg259_1", weight, (CHANNELS,), ((1,),))
    bias_t = _require_f32_tensor("arg260_1", bias, (CHANNELS,), ((1,),))
    _require_shape("_shape_param_0", view_shape, OUTPUT_SHAPE)

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([1280], f32), T([128, 1280, 7, 7], f32, stride=(62720, 1, 8960, 1280)), T([1280], f32), T([1280], f32), T([1280], f32), S([128, 1280]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope BN-affine + ReLU6 + spatial-mean oracle."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu6_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    _bn_relu6_spatial_mean_kernel[(triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        batch=BATCH,
        channels=CHANNELS,
        width=WIDTH,
        hw=HW,
        eps=EPS,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate deterministic output values while requiring matching NaN masks."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    if expected.shape != actual.shape:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False

    dtype_ok = expected.dtype == actual.dtype
    layout_ok = expected.stride() == actual.stride()

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_mask_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan & ~actual_nan

    if finite.any():
        finite_expected = expected_f32[finite]
        finite_actual = actual_f32[finite]
        max_diff = (finite_expected - finite_actual).abs().max().item()
        values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = dtype_ok and layout_ok and nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} "
        f"oracle_dtype={actual.dtype} expected_stride={expected.stride()} "
        f"oracle_stride={actual.stride()} nan_mask={'PASS' if nan_mask_ok else 'FAIL'} "
        f"max_finite_diff={max_diff:.2e})"
    )
    return ok


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
        ok = _run_check(
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
