"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG dual inference-BatchNorm branch sum, NaN-preserving ReLU, 7x7 spatial mean, and final contiguous f32 `[128,1408]` view in one batch/channel-tiled Triton reduction for both recorded NCHW and channels-last strides, whereas Inductor lowers the decomposed broadcast pointwise chain plus small spatial mean through its generic fused reduction schedule with higher constant overhead; Inductor cannot do this today because norm-template canonicalization does not provide a guarded dual-BN-affine branch-add plus activation plus fixed-spatial-mean schedule that reuses per-channel parameters across a batch/channel tile and writes the final view layout directly; the fix is SCHEDULER_FUSION: add a benchmark-gated small-spatial reduction template for paired inference-BN branches followed by ReLU and direct output-layout emission."""
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
CHANNELS = 1408
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CONTIGUOUS_INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
CHANNELS_LAST_INPUT_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)

BLOCK_B = 4
BLOCK_C = 8
BLOCK_HW = 64
CLASSIFICATION = "SCHEDULER_FUSION"
TRUE_FLOOR = True

if triton is not None:

    @triton.jit
    def _dual_bn_relu_spatial_mean_kernel(
        mean0_ptr,
        x0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        x1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        out_ptr,
        x0_stride_n: tl.constexpr,
        x0_stride_c: tl.constexpr,
        x0_stride_h: tl.constexpr,
        x0_stride_w: tl.constexpr,
        x1_stride_n: tl.constexpr,
        x1_stride_c: tl.constexpr,
        x1_stride_h: tl.constexpr,
        x1_stride_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_B_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        batch_offsets = tl.program_id(0) * BLOCK_B_ + tl.arange(0, BLOCK_B_)
        channel_offsets = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        h_offsets = hw_offsets // 7
        w_offsets = hw_offsets - h_offsets * 7

        valid_b = batch_offsets < 128
        valid_c = channel_offsets < 1408
        valid_hw = hw_offsets < 49
        valid_bc = valid_b[:, None] & valid_c[None, :]
        valid = valid_bc[:, :, None] & valid_hw[None, None, :]

        x0_offsets = (
            batch_offsets[:, None, None] * x0_stride_n
            + channel_offsets[None, :, None] * x0_stride_c
            + h_offsets[None, None, :] * x0_stride_h
            + w_offsets[None, None, :] * x0_stride_w
        )
        x1_offsets = (
            batch_offsets[:, None, None] * x1_stride_n
            + channel_offsets[None, :, None] * x1_stride_c
            + h_offsets[None, None, :] * x1_stride_h
            + w_offsets[None, None, :] * x1_stride_w
        )
        x0 = tl.load(x0_ptr + x0_offsets, mask=valid, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + x1_offsets, mask=valid, other=0.0).to(tl.float32)

        mean0 = tl.load(mean0_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)
        var0 = tl.load(var0_ptr + channel_offsets, mask=valid_c, other=1.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)

        mean1 = tl.load(mean1_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + channel_offsets, mask=valid_c, other=1.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)

        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)
        y0 = (x0 - mean0[None, :, None]) * invstd0[None, :, None]
        y1 = (x1 - mean1[None, :, None]) * invstd1[None, :, None]
        y0 = y0 * weight0[None, :, None] + bias0[None, :, None]
        y1 = y1 * weight1[None, :, None] + bias1[None, :, None]

        summed = y0 + y1
        relu = tl.where(summed != summed, summed, tl.maximum(summed, 0.0))
        reduced = tl.sum(tl.where(valid_hw[None, None, :], relu, 0.0), axis=2) * (1.0 / 49.0)

        out_offsets = batch_offsets[:, None] * 1408 + channel_offsets[None, :]
        tl.store(out_ptr + out_offsets, reduced, mask=valid_bc)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    allowed_strides: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) not in allowed_strides:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected one of {allowed_strides}"
        )
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        shape_param,
    ) = inputs

    allowed_x_strides = (CONTIGUOUS_INPUT_STRIDE, CHANNELS_LAST_INPUT_STRIDE)
    mean0_t = _require_f32_tensor("arg280_1", mean0, VECTOR_SHAPE, (VECTOR_STRIDE,))
    x0_t = _require_f32_tensor("convolution_42", x0, INPUT_SHAPE, allowed_x_strides)
    var0_t = _require_f32_tensor("arg281_1", var0, VECTOR_SHAPE, (VECTOR_STRIDE,))
    weight0_t = _require_f32_tensor("arg282_1", weight0, VECTOR_SHAPE, (VECTOR_STRIDE,))
    bias0_t = _require_f32_tensor("arg283_1", bias0, VECTOR_SHAPE, (VECTOR_STRIDE,))
    mean1_t = _require_f32_tensor("arg285_1", mean1, VECTOR_SHAPE, (VECTOR_STRIDE,))
    x1_t = _require_f32_tensor("convolution_43", x1, INPUT_SHAPE, allowed_x_strides)
    var1_t = _require_f32_tensor("arg286_1", var1, VECTOR_SHAPE, (VECTOR_STRIDE,))
    weight1_t = _require_f32_tensor("arg287_1", weight1, VECTOR_SHAPE, (VECTOR_STRIDE,))
    bias1_t = _require_f32_tensor("arg288_1", bias1, VECTOR_SHAPE, (VECTOR_STRIDE,))
    _require_shape("_shape_param_0", shape_param, OUTPUT_SHAPE)

    tensors = (mean0_t, x0_t, var0_t, weight0_t, bias0_t, mean1_t, x1_t, var1_t, weight1_t, bias1_t)
    device = x0_t.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return tensors


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
        raise RuntimeError("Triton is required for oracle_dual_bn_relu_spatial_mean.py")

    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x0.device,
        dtype=torch.float32,
    )
    _dual_bn_relu_spatial_mean_kernel[
        (triton.cdiv(BATCH, BLOCK_B), triton.cdiv(CHANNELS, BLOCK_C))
    ](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        output,
        x0_stride_n=x0.stride(0),
        x0_stride_c=x0.stride(1),
        x0_stride_h=x0.stride(2),
        x0_stride_w=x0.stride(3),
        x1_stride_n=x1.stride(0),
        x1_stride_c=x1.stride(1),
        x1_stride_h=x1.stride(2),
        x1_stride_w=x1.stride(3),
        eps=EPS,
        BLOCK_B_=BLOCK_B,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return output


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate deterministic values and exact output metadata, including NaN masks."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} "
                f"eager={expected.dtype}"
            )
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
            all_pass = False
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            all_pass = all_pass and bool(ok)
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_mask_ok = torch.equal(expected_nan, actual_nan)
        finite = ~expected_nan & ~actual_nan
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
            values_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            values_ok = True
        ok = nan_mask_ok and values_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={expected.stride()} nan_mask={'PASS' if nan_mask_ok else 'FAIL'} "
            f"nan_count={int(expected_nan.sum().item())} max_finite_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


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
