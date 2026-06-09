"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV2 per-channel BN-inference affine plus ReLU6 pointwise scope in one Triton kernel that tiles NCHW by channel and spatial blocks, reusing each channel's mean/variance/weight/bias loads across a spatial tile while preserving the fp32 broadcast math and final fp16 output, whereas Inductor fuses the decomposed unsqueeze/sub/rsqrt/mul/add/clamp graph through its generic pointwise schedule and does not reliably choose this channel-broadcast reuse layout; Inductor cannot do this today because the scheduler/codegen path treats the broadcasted channel vectors as ordinary elementwise producers instead of selecting a layout-aware per-channel affine template for NCHW inference batchnorm epilogues; the fix is SCHEDULER_FUSION: add a guarded pointwise scheduling/template path for per-channel BN affine plus activation that tiles by channel and spatial extent and reuses channel parameters within each program."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
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


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
        ],
        key=["C", "HW"],
    )
    @triton.jit
    def _bn_relu6_kernel(
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

        invstd = 1.0 / tl.sqrt(var + 1.0e-5)
        y = (x - mean[:, None]) * invstd[:, None] * weight[:, None] + bias[:, None]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        y = tl.where(y != y, y, tl.minimum(y, 6.0))
        tl.store(out_ptr + offsets, y, mask=mask)


def _require_f16_vector(name: str, value: Any, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu6_pointwise.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_49 must be a tensor, got {type(x)!r}")
    if x.dim() != 4:
        raise ValueError(f"convolution_49 must be rank 4, got shape={tuple(x.shape)}")
    if x.dtype != torch.float16:
        raise TypeError(f"convolution_49 has dtype {x.dtype}, expected torch.float16")
    if not x.is_cuda:
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous NCHW layout")

    _, channels, _, _ = x.shape
    channel_shape = (channels,)
    mean_t = _require_f16_vector("arg247_1", mean, channel_shape, x.device)
    var_t = _require_f16_vector("arg248_1", var, channel_shape, x.device)
    weight_t = _require_f16_vector("arg249_1", weight, channel_shape, x.device)
    bias_t = _require_f16_vector("arg250_1", bias, channel_shape, x.device)
    return mean_t, x, var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([960], f16), T([128, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward BN-inference plus ReLU6 pointwise scope."""
    mean, x, var, weight, bias = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    hw = height * width
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float16,
    )

    grid = lambda meta: (
        batch,
        triton.cdiv(channels, meta["BLOCK_C"]),
        triton.cdiv(hw, meta["BLOCK_HW"]),
    )
    _bn_relu6_kernel[grid](
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


def _check_oracle(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate the deterministic full-scope output, including matching NaN masks."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = tuple(expected.shape) == tuple(actual.shape)
    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()

    expected_f32 = expected.float()
    actual_f32 = actual.float()
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
        f"  output 0 values: {'PASS' if values_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} dtype={actual.dtype} max_finite_diff={max_diff:.2e})"
    )
    print(
        f"  output 0 layout: {'PASS' if shape_ok and dtype_ok and stride_ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    print(
        f"  output 0 NaNs: {'PASS' if nan_ok else 'FAIL'} "
        f"(expected_nan={int(expected_nan.sum().item())}, "
        f"oracle_nan={int(actual_nan.sum().item())})"
    )
    return values_ok and shape_ok and dtype_ok and stride_ok and nan_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle(
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
