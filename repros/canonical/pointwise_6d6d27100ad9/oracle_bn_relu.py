"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full DenseNet `cat -> BN-inference affine -> ReLU` graph in one channel-blocked Triton kernel, reading each output value from either the 16-channel convolution input or the 152-channel cat input and loading the running mean, running variance, affine scale, and affine bias once per batch-channel tile before broadcasting across the 4x4 spatial block, whereas Inductor currently emits a generic flat pointwise kernel that inlines the cat through a per-element channel predicate but reloads the same per-channel BN parameters and recomputes sqrt/reciprocal for every spatial element; Inductor cannot do this today because the pointwise scheduler has no BN-inference affine-ReLU template over a virtual channel concat that hoists channel-invariant scalar work across the spatial tile; the fix is NEW_PATTERN: add a channel-blocked BN-inference pointwise template that accepts fixed channel-concat sources and computes the per-channel normalization scalars once per tile."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
CONV_CHANNELS = 16
CAT_CHANNELS = 152
CHANNELS = CONV_CHANNELS + CAT_CHANNELS
HEIGHT = 4
WIDTH = 4
SPATIAL = HEIGHT * WIDTH
EPS = 1.0e-5

BLOCK_BATCH = 4
BLOCK_CHANNELS = 16
BLOCK_ROWS = BLOCK_BATCH * BLOCK_CHANNELS
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _cat_bn_relu_kernel(
        conv_ptr,
        cat_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        batch: tl.constexpr,
        conv_channels: tl.constexpr,
        cat_channels: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        eps: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        c_block = tl.program_id(1)

        row_offsets = tl.arange(0, BLOCK_R)
        n_offsets = n_block * BLOCK_N + row_offsets // BLOCK_C
        c_offsets = c_block * BLOCK_C + row_offsets % BLOCK_C
        hw_offsets = tl.arange(0, BLOCK_HW)

        row_mask = (n_offsets < batch) & (c_offsets < channels)
        use_conv = c_offsets < conv_channels

        conv_offsets = (
            (n_offsets[:, None] * conv_channels + c_offsets[:, None]) * spatial
            + hw_offsets[None, :]
        )
        cat_offsets = (
            (n_offsets[:, None] * cat_channels + (c_offsets[:, None] - conv_channels)) * spatial
            + hw_offsets[None, :]
        )
        out_offsets = (
            (n_offsets[:, None] * channels + c_offsets[:, None]) * spatial
            + hw_offsets[None, :]
        )
        value_mask = row_mask[:, None]

        conv_values = tl.load(conv_ptr + conv_offsets, mask=value_mask & use_conv[:, None], other=0.0)
        cat_values = tl.load(cat_ptr + cat_offsets, mask=value_mask & (~use_conv)[:, None], other=0.0)
        x = tl.where(use_conv[:, None], conv_values, cat_values).to(tl.float32)

        mean = tl.load(mean_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=row_mask, other=0.0).to(tl.float32)

        normalized = (x - mean[:, None]) * tl.rsqrt(var[:, None] + eps)
        affine = normalized * weight[:, None] + bias[:, None]
        relu = tl.where(affine != affine, affine, tl.maximum(affine, 0.0))
        tl.store(out_ptr + out_offsets, relu, mask=value_mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_49, cat_21, mean, var, weight, bias = inputs
    conv = _require_tensor(
        "convolution_49",
        convolution_49,
        (BATCH, CONV_CHANNELS, HEIGHT, WIDTH),
        (CONV_CHANNELS * SPATIAL, SPATIAL, WIDTH, 1),
    )
    cat = _require_tensor(
        "cat_21",
        cat_21,
        (BATCH, CAT_CHANNELS, HEIGHT, WIDTH),
        (CAT_CHANNELS * SPATIAL, SPATIAL, WIDTH, 1),
    )
    mean_t = _require_tensor("arg248_1", mean, (CHANNELS,), (1,))
    var_t = _require_tensor("arg249_1", var, (CHANNELS,), (1,))
    weight_t = _require_tensor("arg250_1", weight, (CHANNELS,), (1,))
    bias_t = _require_tensor("arg251_1", bias, (CHANNELS,), (1,))

    device = conv.device
    if any(t.device != device for t in (cat, mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return conv, cat, mean_t, var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([128, 16, 4, 4], f32), T([128, 152, 4, 4], f32), T([168], f32), T([168], f32), T([168], f32), T([168], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward scope."""
    conv, cat, mean, var, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * SPATIAL, SPATIAL, WIDTH, 1),
        device=conv.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(BATCH, BLOCK_BATCH), triton.cdiv(CHANNELS, BLOCK_CHANNELS))
    _cat_bn_relu_kernel[grid](
        conv,
        cat,
        mean,
        var,
        weight,
        bias,
        out,
        batch=BATCH,
        conv_channels=CONV_CHANNELS,
        cat_channels=CAT_CHANNELS,
        channels=CHANNELS,
        spatial=SPATIAL,
        BLOCK_N=BLOCK_BATCH,
        BLOCK_C=BLOCK_CHANNELS,
        BLOCK_R=BLOCK_ROWS,
        BLOCK_HW=SPATIAL,
        eps=EPS,
        num_warps=8,
        num_stages=3,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and expected.stride() == actual.stride()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


def _run_check(instance: torch.nn.Module, inputs: list[Any], *, atol: float, rtol: float) -> bool:
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

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_mask_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan
    if finite.any():
        max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        values_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} max_finite_diff={max_diff:.2e} "
        f"nan_count={int(expected_nan.sum().item())})"
    )
    if expected.dtype != actual.dtype:
        print(f"  output 0: WARNING dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
    return ok


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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
        ok = _check_layout(instance, inputs) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
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
