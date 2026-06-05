"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileNetV3 inference BatchNorm affine, ReLU, and 28x28 spatial mean returned by Repro.forward, including the recorded contiguous and channels-last NCHW input layouts and the keepdim f32 [512,C,1,1] output, in one row-blocked Triton reduction that folds the per-channel affine into scale/shift before reducing, whereas Inductor currently lowers the decomposed sub/sqrt/reciprocal/mul/add/relu/mean graph through a generic reduction that preserves more per-element affine arithmetic; Inductor cannot do this today because its reduction codegen does not canonicalize inference-BN parameters into affine scale/shift inside activation-fed spatial mean reductions; the fix is ALGEBRAIC_ELIMINATION: add BN-inference affine folding for spatial reduction inputs and emit the final keepdim output directly."""
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
BATCH = 512
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_CHANNELS = 128
BLOCK_HW = 1024

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=8, num_stages=3),
        ],
        key=["channels"],
    )
    @triton.jit
    def _bn_relu_spatial_mean_contig_kernel(
        mean_ptr,
        x_ptr,
        variance_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        spatial = tl.arange(0, BLOCK_HW_)
        row_mask = rows < total_rows
        spatial_mask = spatial < 784

        c = rows - (rows // channels) * channels
        x_offsets = rows[:, None] * 784 + spatial[None, :]
        mask = row_mask[:, None] & spatial_mask[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        variance = tl.load(variance_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

        invstd = tl.rsqrt(variance + 1.0e-5)
        scale = invstd * weight
        shift = bias - mean * scale
        y = x * scale[:, None] + shift[:, None]
        relu = tl.where(y < 0.0, 0.0, y)
        reduced = tl.sum(tl.where(spatial_mask[None, :], relu, 0.0), axis=1) * (1.0 / 784.0)

        tl.store(out_ptr + rows, reduced, mask=row_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=8, num_stages=3),
        ],
        key=["channels", "stride_c"],
    )
    @triton.jit
    def _bn_relu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        variance_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        spatial = tl.arange(0, BLOCK_HW_)
        row_mask = rows < total_rows
        spatial_mask = spatial < 784

        n = rows // channels
        c = rows - n * channels
        h = spatial // 28
        w = spatial - h * 28

        x_offsets = (
            n[:, None] * stride_n
            + c[:, None] * stride_c
            + h[None, :] * stride_h
            + w[None, :] * stride_w
        )
        mask = row_mask[:, None] & spatial_mask[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        variance = tl.load(variance_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

        invstd = tl.rsqrt(variance + 1.0e-5)
        scale = invstd * weight
        shift = bias - mean * scale
        y = x * scale[:, None] + shift[:, None]
        relu = tl.where(y < 0.0, 0.0, y)
        reduced = tl.sum(tl.where(spatial_mask[None, :], relu, 0.0), axis=1) * (1.0 / 784.0)

        tl.store(out_ptr + rows, reduced, mask=row_mask)


def _expect_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, variance, weight, bias = inputs
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise TypeError("convolution_20 must be a rank-4 tensor")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    if batch != BATCH or channels not in (72, 120) or (height, width) != (HEIGHT, WIDTH):
        raise ValueError(f"unsupported recorded shape: {tuple(x.shape)}")

    mean_t = _expect_f32_tensor("arg90_1", mean, (channels,))
    x_t = _expect_f32_tensor("convolution_20", x, (batch, channels, height, width))
    variance_t = _expect_f32_tensor("arg91_1", variance, (channels,))
    weight_t = _expect_f32_tensor("arg92_1", weight, (channels,))
    bias_t = _expect_f32_tensor("arg93_1", bias, (channels,))

    if any(tuple(t.stride()) != (1,) for t in (mean_t, variance_t, weight_t, bias_t)):
        raise ValueError("all vector inputs must have stride (1,)")
    if any(t.device != x_t.device for t in (mean_t, variance_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, variance_t, weight_t, bias_t, channels


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
        raise RuntimeError("Triton is required for oracle_bn_relu_spatial_mean.py")

    mean, x, variance, weight, bias, channels = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(BATCH * channels, meta["BLOCK_ROWS"]),)
    if tuple(x.stride()) == (channels * HW, HW, WIDTH, 1):
        _bn_relu_spatial_mean_contig_kernel[grid](
            mean,
            x,
            variance,
            weight,
            bias,
            output,
            total_rows=BATCH * channels,
            channels=channels,
            BLOCK_HW_=BLOCK_HW,
        )
    else:
        _bn_relu_spatial_mean_kernel[grid](
            mean,
            x,
            variance,
            weight,
            bias,
            output,
            total_rows=BATCH * channels,
            channels=channels,
            stride_n=int(x.stride(0)),
            stride_c=int(x.stride(1)),
            stride_h=int(x.stride(2)),
            stride_w=int(x.stride(3)),
            BLOCK_HW_=BLOCK_HW,
        )
    return output


def _normalize_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate full-scope deterministic outputs, including matching NaN masks."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_outputs = _normalize_outputs(expected)
    actual_outputs = _normalize_outputs(actual)
    if len(actual_outputs) != len(expected_outputs):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_outputs)} outputs, "
            f"eager produces {len(expected_outputs)}"
        )
        return False

    all_pass = True
    for index, (expected_item, actual_item) in enumerate(zip(expected_outputs, actual_outputs)):
        if not isinstance(expected_item, torch.Tensor) or not isinstance(actual_item, torch.Tensor):
            ok = expected_item == actual_item
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        shape_ok = expected_item.shape == actual_item.shape
        stride_ok = expected_item.stride() == actual_item.stride()
        dtype_ok = expected_item.dtype == actual_item.dtype
        if not shape_ok or not stride_ok or not dtype_ok:
            print(
                f"  output {index}: SCOPE_MISMATCH "
                f"shape oracle={list(actual_item.shape)} eager={list(expected_item.shape)} "
                f"stride oracle={actual_item.stride()} eager={expected_item.stride()} "
                f"dtype oracle={actual_item.dtype} eager={expected_item.dtype}"
            )
            all_pass = False
            continue

        if not expected_item.is_floating_point():
            ok = torch.equal(expected_item, actual_item)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected_item.dtype})")
            all_pass = all_pass and bool(ok)
            continue

        expected_f32 = expected_item.float()
        actual_f32 = actual_item.float()
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

        ok = nan_mask_ok and values_ok
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_item.shape)} dtype={expected_item.dtype} "
            f"stride={expected_item.stride()} nan_mask={'PASS' if nan_mask_ok else 'FAIL'} "
            f"max_finite_diff={max_diff:.2e})"
        )
        all_pass = all_pass and ok

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
