"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 BN-inference affine, hard-swish producer, fp16 rounding boundary, and spatial mean output in one Triton reduction over each batch/channel tile, whereas Inductor currently lowers the same decomposed sub/sqrt/reciprocal/affine/add/clamp/mul/div/cast/mean graph through its generic fused reduction schedule; Inductor cannot do this today because the norm-template scheduler does not expose a reusable guarded template for BN-affine plus hard-swish feeding a fixed small spatial mean while preserving the explicit fp16 cast before reduction; the fix is SCHEDULER_FUSION: add a benchmark-gated BN-inference activation-spatial-mean template that folds per-channel scale/shift and hard-swish into the reduction kernel and falls back to the generic persistent schedule when it wins."""
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


EPS = 1.0e-3
BLOCK_ROWS = 16

if triton is not None:

    @triton.jit
    def _bn_hardswish_spatial_mean_kernel(
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

        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw_size
        x_offsets = row_offsets[:, None] * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid_rows[:, None] & valid_hw[None, :], other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        affine = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
        relu6 = affine + 3.0
        relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
        relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
        hardswish = affine * relu6 * (1.0 / 6.0)
        hardswish_h = hardswish.to(tl.float16)
        reduced = (
            tl.sum(tl.where(valid_hw[None, :], hardswish_h.to(tl.float32), 0.0), axis=1)
            / hw_size
        )

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise TypeError("convolution_57 must be a rank-4 tensor")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    hw_size = height * width
    if batch != 256 or channels not in (480, 672, 960) or (height, width) not in ((7, 7), (14, 14)):
        raise ValueError(f"unsupported MobileNetV3 shape: {tuple(x.shape)}")

    expected_stride = (channels * hw_size, hw_size, width, 1)
    mean_t = _require_f16_tensor("arg245_1", mean, (channels,), (1,))
    x_t = _require_f16_tensor("convolution_57", x, (batch, channels, height, width), expected_stride)
    var_t = _require_f16_tensor("arg246_1", var, (channels,), (1,))
    weight_t = _require_f16_tensor("arg247_1", weight, (channels,), (1,))
    bias_t = _require_f16_tensor("arg248_1", bias, (channels,), (1,))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t, batch, channels, height, width


@oracle_impl(hardware="H100", shapes="(T([960], f16), T([256, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16))")
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
        raise RuntimeError("Triton is required for oracle_mobilenet_bn_hardswish_spatial_mean.py")

    mean, x, var, weight, bias, batch, channels, height, width = _validate_inputs(inputs)
    hw_size = height * width
    block_hw = triton.next_power_of_2(hw_size)
    output = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float16,
    )

    _bn_hardswish_spatial_mean_kernel[(triton.cdiv(batch * channels, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        total_rows=batch * channels,
        channels=channels,
        hw_size=hw_size,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=block_hw,
        eps=EPS,
        num_warps=1,
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
    """Validate deterministic full-scope output, preserving NaN locations."""
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
