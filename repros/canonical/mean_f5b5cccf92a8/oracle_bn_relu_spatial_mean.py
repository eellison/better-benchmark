"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full MobileNetV3 fp16 BN-inference affine, explicit fp16 rounding, NaN-preserving ReLU, and 28x28 spatial mean directly into the final fp16 `[256,C,1,1]` keepdim output for the recorded C=120/C=72 shapes, while hoisting per-channel scale/shift and algebraically skipping activation loads for rows whose runtime `variance + 0.001` makes the affine result NaN for every spatial element; Inductor lowers the broadcast normalization, activation, cast, and spatial reduction through generic norm-template reduction code that still reads the full activation tile for those NaN-dominated rows; Inductor cannot do this today because its simplifier/reduction scheduler does not derive and guard value-dependent row results that are independent of the reduction input while preserving fp16 rounding and NaN behavior; the fix is ALGEBRAIC_ELIMINATION: add guarded NaN-dominance elimination for inference-BN spatial reductions plus reusable per-channel affine hoisting before lowering the reduction."""
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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 256
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
EPS = 1.0e-3
BLOCK_HW = 1024
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


if triton is not None:

    @triton.jit
    def _affine_precompute_kernel(
        mean_ptr,
        variance_ptr,
        weight_ptr,
        bias_ptr,
        affine_ptr,
        channels: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        eps: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        mask = offsets < channels

        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.load(variance_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scale = (1.0 / tl.sqrt(variance + eps)) * weight
        shift = bias - mean * scale
        tl.store(affine_ptr + offsets, scale, mask=mask)
        tl.store(affine_ptr + channels + offsets, shift, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=8, num_stages=3),
        ],
        key=["channels"],
    )
    @triton.jit
    def _bn_relu_spatial_mean_kernel(
        affine_ptr,
        x_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw = tl.arange(0, BLOCK_HW_)
        row_mask = rows < total_rows
        hw_mask = hw < 784

        batch = rows // channels
        channel = rows - batch * channels
        x_offsets = rows[:, None] * 784 + hw[None, :]
        valid = row_mask[:, None] & hw_mask[None, :]

        scale = tl.load(affine_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        shift = tl.load(affine_ptr + channels + channel, mask=row_mask, other=0.0).to(tl.float32)
        nan_row = scale != scale

        x = tl.load(x_ptr + x_offsets, mask=valid & (~nan_row[:, None]), other=0.0).to(tl.float32)
        y = x * scale[:, None] + shift[:, None]
        y_h = y.to(tl.float16)
        zero_h = tl.full((BLOCK_ROWS, BLOCK_HW_), 0.0, tl.float16)
        relu_h = tl.where(y_h < zero_h, zero_h, y_h)
        reduced = tl.sum(tl.where(valid, relu_h.to(tl.float32), 0.0), axis=1) * (1.0 / 784.0)
        tl.store(out_ptr + rows, reduced, mask=row_mask)


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
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, variance, weight, bias = inputs
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise TypeError("convolution_20 must be a 4D tensor")

    batch, channels, height, width = x.shape
    if int(batch) != BATCH:
        raise ValueError(f"convolution_20 has batch {int(batch)}, expected {BATCH}")
    if (int(height), int(width)) != (HEIGHT, WIDTH):
        raise ValueError(f"convolution_20 has spatial shape {(int(height), int(width))}, expected {(HEIGHT, WIDTH)}")
    if int(channels) not in (72, 120):
        raise ValueError(f"unsupported recorded channel count {int(channels)}")

    channels = int(channels)
    vector_shape = (channels,)
    vector_stride = (1,)
    image_shape = (BATCH, channels, HEIGHT, WIDTH)
    image_stride = (channels * HW, HW, WIDTH, 1)

    mean_t = _require_f16_tensor("arg90_1", mean, vector_shape, vector_stride)
    x_t = _require_f16_tensor("convolution_20", x, image_shape, image_stride)
    variance_t = _require_f16_tensor("arg91_1", variance, vector_shape, vector_stride)
    weight_t = _require_f16_tensor("arg92_1", weight, vector_shape, vector_stride)
    bias_t = _require_f16_tensor("arg93_1", bias, vector_shape, vector_stride)

    if any(t.device != x_t.device for t in (mean_t, variance_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, variance_t, weight_t, bias_t, channels


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu_spatial_mean.py")

    mean, x, variance, weight, bias, channels = _validate_inputs(inputs)
    affine = torch.empty_strided(
        (2, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        (BATCH, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float16,
    )
    _affine_precompute_kernel[(triton.cdiv(channels, 256),)](
        mean,
        variance,
        weight,
        bias,
        affine,
        channels=channels,
        BLOCK_CHANNELS=256,
        eps=EPS,
        num_warps=4,
        num_stages=3,
    )
    grid = lambda meta: (triton.cdiv(BATCH * channels, meta["BLOCK_ROWS"]),)
    _bn_relu_spatial_mean_kernel[grid](
        affine,
        x,
        output,
        total_rows=BATCH * channels,
        channels=channels,
        BLOCK_HW_=BLOCK_HW,
    )
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate deterministic full-scope output, including matching NaN masks."""
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
    if expected.stride() != actual.stride():
        print(
            f"  output 0: SCOPE_MISMATCH stride oracle={actual.stride()} "
            f"eager={expected.stride()}"
        )
        return False

    dtype_ok = expected.dtype == actual.dtype
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

    ok = dtype_ok and nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
    )
    if not dtype_ok:
        print(f"  output 0: WARNING dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
    if not nan_mask_ok:
        print(
            f"  output 0: NaN mask mismatch "
            f"(expected_nan={int(expected_nan.sum().item())}, oracle_nan={int(actual_nan.sum().item())})"
        )
    return ok


# --- CLI entry point ---
def main() -> None:
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
            print(f"classification: {CLASSIFICATION}")


if __name__ == "__main__":
    main()
