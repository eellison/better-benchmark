"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet BN-inference affine, ReLU, fixed 7x7 spatial mean, and returned as_strided `[512,960,1,1]` view in one Triton reduction kernel that reuses per-channel parameters across each spatial tile, whereas tuned Inductor already reaches the same CUDAGraph-measured floor for this full norm-template-canonicalization scope; Inductor cannot materially improve this local repro with scheduler fusion, scatter-reduce, split-K, algebraic elimination, or a new narrow pattern because the remaining work is dominated by required activation reads, channel-parameter reads, one small spatial reduction, ReLU math, and output stores; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader normalization-template, memory-traffic, or launch-overhead work moves both paths together."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
EPS = 1.0e-5

X_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
PARAM_SHAPE = (CHANNELS,)
CONTIGUOUS_X_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
CHANNELS_LAST_X_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
OUTPUT_BASE_SHAPE = (BATCH, CHANNELS)
OUTPUT_BASE_STRIDE = (CHANNELS, 1)
OUTPUT_VIEW_SHAPE = (BATCH, CHANNELS, 1, 1)
OUTPUT_VIEW_STRIDE = (CHANNELS, 1, CHANNELS, CHANNELS)
BLOCK_HW = 64
CLASSIFICATION = "BANDWIDTH_BOUND"


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 64}, num_warps=8, num_stages=3),
        ],
        key=["n_rows", "x_stride_c"],
    )
    @triton.jit
    def _bn_relu_spatial_mean_kernel(
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
        n_rows: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        spatial = tl.arange(0, BLOCK_HW_)
        valid_rows = rows < n_rows
        valid_spatial = spatial < hw

        n = rows // channels
        c = rows - n * channels
        h = spatial // width
        w = spatial - h * width
        x_offsets = (
            n[:, None] * x_stride_n
            + c[:, None] * x_stride_c
            + h[None, :] * x_stride_h
            + w[None, :] * x_stride_w
        )
        mask = valid_rows[:, None] & valid_spatial[None, :]

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=valid_rows, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        value = (x - mean[:, None]) * invstd[:, None]
        value = value * weight[:, None] + bias[:, None]
        relu = tl.where(value < 0.0, 0.0, value)
        relu = tl.where(valid_spatial[None, :], relu, 0.0)
        reduced = tl.sum(relu, axis=1) * (1.0 / 49.0)
        tl.store(out_ptr + rows, reduced, mask=valid_rows)


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


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    mean_t = _require_f32_tensor("arg425_1", mean, PARAM_SHAPE, ((1,),))
    x_t = _require_f32_tensor(
        "convolution_93",
        x,
        X_SHAPE,
        (CONTIGUOUS_X_STRIDE, CHANNELS_LAST_X_STRIDE),
    )
    var_t = _require_f32_tensor("arg426_1", var, PARAM_SHAPE, ((1,),))
    weight_t = _require_f32_tensor("arg427_1", weight, PARAM_SHAPE, ((1,),))
    bias_t = _require_f32_tensor("arg428_1", bias, PARAM_SHAPE, ((1,),))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, var_t, weight_t, bias_t


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope BN-affine + ReLU + spatial-mean oracle.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    output_base = torch.empty_strided(
        OUTPUT_BASE_SHAPE,
        OUTPUT_BASE_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _bn_relu_spatial_mean_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output_base,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        n_rows=ROWS,
        channels=CHANNELS,
        width=WIDTH,
        hw=HW,
        eps=EPS,
        BLOCK_HW_=BLOCK_HW,
    )
    return output_base.as_strided(OUTPUT_VIEW_SHAPE, OUTPUT_VIEW_STRIDE)


def _check_with_nan_masks(
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
    stride_ok = expected.stride() == actual.stride()
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

    ok = dtype_ok and stride_ok and nan_mask_ok and values_ok
    status = "PASS" if ok else "FAIL"
    print(
        f"  output 0: {status} (shape={list(expected.shape)} "
        f"stride={tuple(expected.stride())} dtype={expected.dtype} "
        f"max_diff={max_diff:.2e} nan_count={expected_nan.sum().item()})"
    )
    if not dtype_ok:
        print(f"    dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
    if not stride_ok:
        print(f"    stride mismatch oracle={tuple(actual.stride())} eager={tuple(expected.stride())}")
    if not nan_mask_ok:
        print("    NaN mask mismatch")
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
        ok = _check_with_nan_masks(
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
