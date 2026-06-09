"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet BN-inference affine over fp16 `[64,1152,7,7]`, evaluates SiLU as `x/(exp(-x)+1)`, rounds through fp16 before the spatial 7x7 mean, preserves NaN propagation, and returns the contiguous fp16 `[64,1152,1,1]` output from one Triton reduction kernel, whereas Inductor currently lowers the same full scope to a fused persistent reduction kernel for BN/SILU/spatial mean that is at least as fast as this hand-written specialization; Inductor cannot do this today because the scheduler has no reusable guarded template for a BN-affine activation producer with an explicit fp16 rounding boundary feeding a fixed small spatial mean reduction, only the generic persistent reduction schedule; the fix is SCHEDULER_FUSION: add a benchmark-gated scheduler fusion template for BN-affine plus activation plus small spatial mean that preserves the fp16 cast and NaN semantics, and leave the existing persistent schedule selected when it wins."""
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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        c_offsets = row_offsets - (row_offsets // channels) * channels
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_hw = hw_offsets < hw_size
        x_offsets = row_offsets[:, None] * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid_hw[None, :], other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets).to(tl.float32)
        var = tl.load(var_ptr + c_offsets).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        affine = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
        silu = affine / (tl.exp(-affine) + 1.0)
        silu_h = silu.to(tl.float16)
        reduced = tl.sum(tl.where(valid_hw[None, :], silu_h.to(tl.float32), 0.0), axis=1) / hw_size

        tl.store(out_ptr + row_offsets, reduced)


BATCH = 64
CHANNELS = 1152
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_ROWS = 16
BLOCK_HW = 64
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = False


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    x_t = _require_f16_tensor(
        "convolution_76",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    mean_t = _require_f16_tensor("arg292_1", mean, (CHANNELS,), (1,))
    var_t = _require_f16_tensor("arg293_1", var, (CHANNELS,), (1,))
    weight_t = _require_f16_tensor("arg294_1", weight, (CHANNELS,), (1,))
    bias_t = _require_f16_tensor("arg295_1", bias, (CHANNELS,), (1,))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([1152], f16), T([64, 1152, 7, 7], f16), T([1152], f16), T([1152], f16), T([1152], f16))")
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
        raise RuntimeError("Triton is required for oracle_bn_silu_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float16,
    )
    grid = (triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)
    _bn_silu_spatial_mean_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        total_rows=BATCH * CHANNELS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
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
    """Validate the deterministic full-scope output, including matching NaNs."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    layout_ok = expected.stride() == actual.stride()

    if not shape_ok:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False

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

    value_status = "PASS" if values_ok and dtype_ok else "FAIL"
    layout_status = "PASS" if layout_ok else "FAIL"
    nan_status = "PASS" if nan_mask_ok else "FAIL"
    print(
        f"  output 0 values: {value_status} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} "
        f"oracle_dtype={actual.dtype} max_finite_diff={max_diff:.2e})"
    )
    print(
        f"  output 0 layout: {layout_status} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    print(
        f"  output 0 NaNs: {nan_status} "
        f"(expected_nan={int(expected_nan.sum().item())}, "
        f"oracle_nan={int(actual_nan.sum().item())})"
    )
    return values_ok and dtype_ok and layout_ok and nan_mask_ok


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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
            true_floor = result["status"] == "GOOD"
            print(f"classification: {CLASSIFICATION}")
            print(f"true_floor: {'yes' if true_floor else 'no'} ({result['status']})")
            print(f"actionable: {'yes' if ACTIONABLE and true_floor else 'no'}")
            if not true_floor:
                print("diagnosis_only: not_true_floor because compiled Inductor is at least as fast as this full-scope oracle")


if __name__ == "__main__":
    main()
