"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured BN-inference affine, fp16 ReLU, channel-pair sum, and spatial mean return in one Triton kernel that stores only the final fp16 [N, C/2, 1, 1] tensor, whereas tuned Inductor is already within the oracle harness floor band for the same full output contract; Inductor cannot get a confirmed local win here because the remaining cost is dominated by required reads of the fp16 activation and affine vectors, fp16 rounding/ReLU semantics, pairwise channel reduction, spatial accumulation, and launch overhead rather than an exposed materialization gap; the fix is BANDWIDTH_BOUND: record this repro as at floor and only revisit if a broader norm-plus-pooling template beats the current compiled kernel on the exact full scope."""
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

EPS = 1.0e-5


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _block_channels(hw_size: int, half_channels: int) -> int:
    if hw_size <= 256:
        return 8 if half_channels >= 256 else 4
    if hw_size <= 1024:
        return 4
    return 2

if triton is not None:

    @triton.jit
    def _bn_relu_pair_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        half_channels: tl.constexpr,
        hw_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        channel_valid = c_offsets < half_channels
        hw_valid = hw_offsets < hw_size

        c1_offsets = c_offsets + half_channels

        mean0 = tl.load(mean_ptr + c_offsets, mask=channel_valid, other=0.0).to(tl.float32)
        mean1 = tl.load(mean_ptr + c1_offsets, mask=channel_valid, other=0.0).to(tl.float32)
        var0 = tl.load(var_ptr + c_offsets, mask=channel_valid, other=1.0).to(tl.float32)
        var1 = tl.load(var_ptr + c1_offsets, mask=channel_valid, other=1.0).to(tl.float32)
        weight0 = tl.load(weight_ptr + c_offsets, mask=channel_valid, other=0.0).to(tl.float32)
        weight1 = tl.load(weight_ptr + c1_offsets, mask=channel_valid, other=0.0).to(tl.float32)
        bias0 = tl.load(bias_ptr + c_offsets, mask=channel_valid, other=0.0).to(tl.float32)
        bias1 = tl.load(bias_ptr + c1_offsets, mask=channel_valid, other=0.0).to(tl.float32)
        inv0 = tl.rsqrt(var0 + eps)
        inv1 = tl.rsqrt(var1 + eps)
        scale0 = inv0 * weight0
        scale1 = inv1 * weight1
        shift0 = bias0 - mean0 * scale0
        shift1 = bias1 - mean1 * scale1

        base = batch * channels * hw_size
        x0_offsets = base + c_offsets[:, None] * hw_size + hw_offsets[None, :]
        x1_offsets = base + c1_offsets[:, None] * hw_size + hw_offsets[None, :]
        valid = channel_valid[:, None] & hw_valid[None, :]
        x0 = tl.load(x_ptr + x0_offsets, mask=valid, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x1_offsets, mask=valid, other=0.0).to(tl.float32)

        y0 = x0 * scale0[:, None] + shift0[:, None]
        y1 = x1 * scale1[:, None] + shift1[:, None]

        y0_h = y0.to(tl.float16)
        y1_h = y1.to(tl.float16)
        y0_relu = tl.where(y0_h != y0_h, y0_h, tl.maximum(y0_h, 0.0)).to(tl.float16)
        y1_relu = tl.where(y1_h != y1_h, y1_h, tl.maximum(y1_h, 0.0)).to(tl.float16)
        pair = (y0_relu + y1_relu).to(tl.float16)
        pair_f32 = tl.where(valid, pair.to(tl.float32), 0.0)
        reduced = tl.sum(pair_f32, axis=1) * (1.0 / hw_size)
        tl.store(out_ptr + batch * half_channels + c_offsets, reduced, mask=channel_valid)


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_22 must be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"convolution_22 must be 4D, got shape {tuple(x.shape)}")

    batch, channels, height, width = tuple(x.shape)
    if channels % 2 != 0:
        raise ValueError(f"convolution_22 channel dimension must be even, got {channels}")
    half_channels = channels // 2
    hw_size = height * width
    expected_x_stride = (channels * hw_size, hw_size, width, 1)

    x_t = _require_f16_tensor(
        "convolution_22",
        x,
        (batch, channels, height, width),
        expected_x_stride,
    )
    mean_t = _require_f16_tensor("arg106_1", mean, (channels,), (1,))
    var_t = _require_f16_tensor("arg107_1", var, (channels,), (1,))
    weight_t = _require_f16_tensor("arg108_1", weight, (channels,), (1,))
    bias_t = _require_f16_tensor("arg109_1", bias, (channels,), (1,))

    if list(shape_param) != [batch, 2, half_channels, height, width]:
        raise ValueError(
            f"unexpected view shape parameter: {shape_param!r}, "
            f"expected {[batch, 2, half_channels, height, width]}"
        )
    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t


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
        raise RuntimeError("Triton is required for oracle_bn_pair_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    batch, channels, height, width = tuple(x.shape)
    half_channels = channels // 2
    hw_size = height * width
    output = torch.empty_strided(
        (batch, half_channels, 1, 1),
        (half_channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float16,
    )
    block_hw = _next_power_of_2(hw_size)
    block_c = _block_channels(hw_size, half_channels)
    grid = (batch, triton.cdiv(half_channels, block_c))
    _bn_relu_pair_spatial_mean_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        channels=channels,
        half_channels=half_channels,
        hw_size=hw_size,
        eps=EPS,
        BLOCK_C=block_c,
        BLOCK_HW=block_hw,
        num_warps=4 if block_hw <= 1024 else 8,
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
    oracle_fn: Any,
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_fn(inputs)
        torch.cuda.synchronize()

    expected_outputs = _normalize_outputs(expected)
    actual_outputs = _normalize_outputs(actual)
    if len(expected_outputs) != len(actual_outputs):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_outputs)} outputs, "
            f"eager produces {len(expected_outputs)}"
        )
        return False

    ok_all = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_outputs, actual_outputs)):
        metadata_ok = True
        if expected_tensor.shape != actual_tensor.shape:
            print(
                f"  output {index}: SCOPE_MISMATCH shape oracle={list(actual_tensor.shape)} "
                f"eager={list(expected_tensor.shape)}"
            )
            metadata_ok = False
        if expected_tensor.dtype != actual_tensor.dtype:
            print(
                f"  output {index}: SCOPE_MISMATCH dtype oracle={actual_tensor.dtype} "
                f"eager={expected_tensor.dtype}"
            )
            metadata_ok = False
        if expected_tensor.stride() != actual_tensor.stride():
            print(
                f"  output {index}: SCOPE_MISMATCH stride oracle={actual_tensor.stride()} "
                f"eager={expected_tensor.stride()}"
            )
            metadata_ok = False
        if not metadata_ok:
            ok_all = False
            continue

        if expected_tensor.is_floating_point():
            expected_f32 = expected_tensor.float()
            actual_f32 = actual_tensor.float()
            expected_nan = torch.isnan(expected_f32)
            actual_nan = torch.isnan(actual_f32)
            nan_mask_ok = torch.equal(expected_nan, actual_nan)
            finite = ~expected_nan
            if finite.any():
                max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
                values_ok = torch.allclose(
                    expected_f32[finite],
                    actual_f32[finite],
                    atol=atol,
                    rtol=rtol,
                )
            else:
                max_diff = 0.0
                values_ok = True
            ok = nan_mask_ok and values_ok
            print(
                f"  output {index}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
                f"stride={expected_tensor.stride()} max_finite_diff={max_diff:.2e} "
                f"nan_count={int(expected_nan.sum().item())})"
            )
        else:
            ok = torch.equal(expected_tensor, actual_tensor)
            print(
                f"  output {index}: {'PASS' if ok else 'FAIL'} "
                f"(exact, shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype})"
            )
        ok_all = ok_all and bool(ok)

    return bool(ok_all)


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
            oracle_forward,
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
