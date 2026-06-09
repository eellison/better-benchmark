"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full GhostNet BN-inference affine, ReLU, virtual channel cat with the existing ReLU branch, and spatial mean directly into the final f32[N,2C,1,1] output, whereas Inductor lowers the cat and reduction through generic scheduling instead of treating the cat as a virtual producer feeding the mean; Inductor cannot do this today because its scheduler does not inline fixed channel-concat sources through a following reduction while also canonicalizing the BN affine producer; the fix is SCHEDULER_FUSION: allow channel cat producers to be virtualized inside reduction templates and sink the BN/ReLU producer into the same final-output loop nest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


EPS = 1.0e-5
BLOCK_CHANNELS = 256
BLOCK_ROWS = 8


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _affine_precompute_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        affine_ptr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        mask = offsets < channels

        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scale = tl.rsqrt(var + eps) * weight
        shift = bias - mean * scale
        tl.store(affine_ptr + offsets, scale, mask=mask)
        tl.store(affine_ptr + channels + offsets, shift, mask=mask)

    @triton.jit
    def _virtual_cat_bn_spatial_mean_kernel(
        x_ptr,
        relu_ptr,
        affine_ptr,
        out_ptr,
        total_rows,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        relu_stride_n: tl.constexpr,
        relu_stride_c: tl.constexpr,
        relu_stride_h: tl.constexpr,
        relu_stride_w: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        out_channels: tl.constexpr = channels * 2
        hw: tl.constexpr = height * width
        batch_offsets = row_offsets // out_channels
        out_channel_offsets = row_offsets - batch_offsets * out_channels
        channel_offsets = out_channel_offsets - tl.where(out_channel_offsets >= channels, channels, 0)

        h_offsets = hw_offsets // width
        w_offsets = hw_offsets - h_offsets * width
        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw
        valid = valid_rows[:, None] & valid_hw[None, :]

        relu_load_offsets = (
            batch_offsets[:, None] * relu_stride_n
            + channel_offsets[:, None] * relu_stride_c
            + h_offsets[None, :] * relu_stride_h
            + w_offsets[None, :] * relu_stride_w
        )
        x_load_offsets = (
            batch_offsets[:, None] * x_stride_n
            + channel_offsets[:, None] * x_stride_c
            + h_offsets[None, :] * x_stride_h
            + w_offsets[None, :] * x_stride_w
        )

        use_bn_branch = out_channel_offsets >= channels
        relu_values = tl.load(relu_ptr + relu_load_offsets, mask=valid, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + x_load_offsets, mask=valid, other=0.0).to(tl.float32)
        scale = tl.load(affine_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        shift = tl.load(affine_ptr + channels + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        y = x * scale[:, None] + shift[:, None]
        zero = tl.full((BLOCK_ROWS_, BLOCK_HW_), 0.0, tl.float32)
        bn_relu = tl.where(y < zero, zero, y)

        values = tl.where(use_bn_branch[:, None], bn_relu, relu_values)
        reduced = tl.sum(tl.where(valid, values, 0.0), axis=1) * (1.0 / hw)
        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_f32_vector(name: str, value: Any, channels: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (channels,):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {(channels,)}")
    if tuple(value.stride()) != (1,):
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected (1,)")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _require_f32_image(
    name: str,
    value: Any,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.stride(-1) != 1 and value.stride(1) != 1:
        raise ValueError(f"{name} must have contiguous spatial or channel-last layout, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, relu = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_88 must be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"convolution_88 must be rank 4, got shape {tuple(x.shape)}")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    image_shape = (batch, channels, height, width)
    mean_t = _require_f32_vector("arg406_1", mean, channels)
    x_t = _require_f32_image("convolution_88", x, image_shape)
    var_t = _require_f32_vector("arg407_1", var, channels)
    weight_t = _require_f32_vector("arg408_1", weight, channels)
    bias_t = _require_f32_vector("arg409_1", bias, channels)
    relu_t = _require_f32_image("relu_37", relu, image_shape)

    device = x_t.device
    if any(t.device != device for t in (mean_t, var_t, weight_t, bias_t, relu_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t, relu_t, batch, channels, height, width


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@oracle_impl(hardware="H100", shapes="(T([480], f32), T([512, 480, 7, 7], f32), T([480], f32), T([480], f32), T([480], f32), T([512, 480, 7, 7], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full virtual-cat BN/ReLU spatial-mean computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_virtual_cat_bn_spatial_mean.py")

    mean, x, var, weight, bias, relu, batch, channels, height, width = _validate_inputs(inputs)
    out_channels = channels * 2
    total_rows = batch * out_channels
    hw = height * width

    affine = torch.empty_strided((2, channels), (channels, 1), device=x.device, dtype=torch.float32)
    output = torch.empty_strided(
        (batch, out_channels, 1, 1),
        (out_channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _affine_precompute_kernel[(triton.cdiv(channels, BLOCK_CHANNELS),)](
        mean,
        var,
        weight,
        bias,
        affine,
        channels=channels,
        eps=EPS,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        num_warps=4,
        num_stages=3,
    )
    _virtual_cat_bn_spatial_mean_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        x,
        relu,
        affine,
        output,
        total_rows,
        channels=channels,
        height=height,
        width=width,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        relu_stride_n=relu.stride(0),
        relu_stride_c=relu.stride(1),
        relu_stride_h=relu.stride(2),
        relu_stride_w=relu.stride(3),
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=_next_power_of_2(hw),
        num_warps=4,
        num_stages=3,
    )
    return output


def _max_finite_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).abs()
    finite = torch.isfinite(diff)
    if not finite.any():
        return 0.0
    return float(diff[finite].max().item())


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate deterministic values while treating matching NaNs as equal."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()
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
    nan_ok = torch.equal(expected_nan, actual_nan)
    finite = ~(expected_nan | actual_nan)
    if finite.any():
        values_ok = torch.allclose(actual_f32[finite], expected_f32[finite], atol=atol, rtol=rtol)
    else:
        values_ok = True
    max_diff = _max_finite_diff(actual, expected)

    ok = shape_ok and dtype_ok and stride_ok and nan_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
    )
    if not dtype_ok:
        print(f"    dtype mismatch: oracle={actual.dtype} eager={expected.dtype}")
    if not stride_ok:
        print(f"    stride mismatch: oracle={actual.stride()} eager={expected.stride()}")
    if not nan_ok:
        print(
            f"    nan mask mismatch: oracle_nan={int(actual_nan.sum().item())} "
            f"eager_nan={int(expected_nan.sum().item())}"
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


if __name__ == "__main__":
    main()
