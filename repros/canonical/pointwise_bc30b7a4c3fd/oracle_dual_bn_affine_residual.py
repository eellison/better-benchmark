"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle folds the full two-stage channelwise BN-inference affine chain with the broadcast residual by precomputing per-channel scale/residual-scale/shift coefficients and streaming the output as a flat storage-order FMA, whereas Inductor currently emits a generic fused pointwise kernel for the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add graph and recomputes both normalization affine chains at element granularity; Inductor cannot do this today because its algebraic simplifier/codegen does not canonicalize chained per-channel BN affine broadcasts plus a residual broadcast into hoisted channel-invariant coefficients before pointwise lowering; the fix is ALGEBRAIC_ELIMINATION: add a guarded BN-affine folding rewrite for serial inference-normalization epilogues that lowers the final loop as `x * scale + residual * residual_scale + shift` while preserving broadcast layouts and NaN behavior."""
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
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
EPS = 1.0e-5


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_bn_coefficients_kernel(
        mean1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        mean2_ptr,
        var2_ptr,
        weight2_ptr,
        bias2_ptr,
        coeff_ptr,
        C: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        mean1 = tl.load(mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean2 = tl.load(mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var2 = tl.load(var2_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        scale1 = (1.0 / tl.sqrt(var1 + eps)) * weight1
        shift1 = bias1 - mean1 * scale1
        scale2 = (1.0 / tl.sqrt(var2 + eps)) * weight2
        x_scale = scale1 * scale2
        shift = (shift1 - mean2) * scale2 + bias2

        tl.store(coeff_ptr + c_offsets, x_scale, mask=c_mask)
        tl.store(coeff_ptr + C + c_offsets, scale2, mask=c_mask)
        tl.store(coeff_ptr + 2 * C + c_offsets, shift, mask=c_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N", "C", "HW", "LAYOUT"],
    )
    @triton.jit
    def _dual_bn_affine_residual_kernel(
        x_ptr,
        residual_ptr,
        coeff_ptr,
        out_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        LAYOUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        if LAYOUT == 0:
            c_offsets = (offsets // HW) % C
        else:
            c_offsets = offsets % C

        residual_offsets = offsets % (C * HW)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        x_scale = tl.load(coeff_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        residual_scale = tl.load(coeff_ptr + C + c_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(coeff_ptr + 2 * C + c_offsets, mask=mask, other=0.0).to(tl.float32)

        out = x * x_scale + residual * residual_scale + shift
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_f32_cuda_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have zero storage_offset")
    return value


def _dense_layout_kind(x: torch.Tensor) -> int:
    if x.ndim != 4:
        raise ValueError(f"convolution_40 must be rank 4, got shape {tuple(x.shape)}")
    _, channels, height, width = x.shape
    nchw_stride = (channels * height * width, height * width, width, 1)
    channels_last_stride = (channels * height * width, 1, width * channels, channels)
    if tuple(x.stride()) == nchw_stride:
        return 0
    if tuple(x.stride()) == channels_last_stride:
        return 1
    raise ValueError(f"unsupported dense input stride for convolution_40: {tuple(x.stride())}")


def _expected_residual_stride(channels: int, height: int, width: int, layout: int) -> tuple[int, int, int, int]:
    if layout == 0:
        return (channels * height * width, height * width, width, 1)
    return (channels * height * width, 1, width * channels, channels)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dual_bn_affine_residual.py")
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        mean1,
        x,
        var1,
        weight1,
        bias1,
        residual,
        mean2,
        var2,
        weight2,
        bias2,
    ) = (
        _require_f32_cuda_tensor(name, value)
        for name, value in zip(
            (
                "arg119_1",
                "convolution_40",
                "arg120_1",
                "arg121_1",
                "arg122_1",
                "arg123_1",
                "arg124_1",
                "arg125_1",
                "arg126_1",
                "arg127_1",
            ),
            inputs,
        )
    )

    layout = _dense_layout_kind(x)
    batch, channels, height, width = x.shape
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"convolution_40 has invalid shape {tuple(x.shape)}")

    if tuple(residual.shape) != (1, channels, height, width):
        raise ValueError(
            f"arg123_1 shape must be {(1, channels, height, width)}, got {tuple(residual.shape)}"
        )
    expected_residual_stride = _expected_residual_stride(channels, height, width, layout)
    if tuple(residual.stride()) != expected_residual_stride:
        raise ValueError(
            f"arg123_1 stride must be {expected_residual_stride}, got {tuple(residual.stride())}"
        )

    for name, tensor in (
        ("arg119_1", mean1),
        ("arg120_1", var1),
        ("arg121_1", weight1),
        ("arg122_1", bias1),
        ("arg124_1", mean2),
        ("arg125_1", var2),
        ("arg126_1", weight2),
        ("arg127_1", bias2),
    ):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} shape must be ({channels},), got {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != (1,):
            raise ValueError(f"{name} must be contiguous, got stride {tuple(tensor.stride())}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same CUDA device as convolution_40")

    if residual.device != x.device:
        raise ValueError("arg123_1 must be on the same CUDA device as convolution_40")

    return mean1, x, var1, weight1, bias1, residual, mean2, var2, weight2, bias2, layout


@oracle_impl(hardware="H100", shapes="(T([768], f32), T([128, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([1, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with folded channelwise affine coefficients."""
    (
        mean1,
        x,
        var1,
        weight1,
        bias1,
        residual,
        mean2,
        var2,
        weight2,
        bias2,
        layout,
    ) = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    hw = height * width

    output = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.float32)
    coeff = torch.empty((3, channels), device=x.device, dtype=torch.float32)
    _dual_bn_coefficients_kernel[(triton.cdiv(channels, 256),)](
        mean1,
        var1,
        weight1,
        bias1,
        mean2,
        var2,
        weight2,
        bias2,
        coeff,
        C=channels,
        eps=EPS,
        BLOCK_C=256,
        num_warps=8,
        num_stages=3,
    )
    grid = lambda meta: (triton.cdiv(batch * channels * hw, meta["BLOCK_N"]),)
    _dual_bn_affine_residual_kernel[grid](
        x,
        residual,
        coeff,
        output,
        N=batch * channels * hw,
        C=channels,
        HW=hw,
        LAYOUT=layout,
    )
    return output


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _check_oracle_nan_aware(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_outputs = _normalize_outputs(expected)
    actual_outputs = _normalize_outputs(actual)
    if len(expected_outputs) != len(actual_outputs):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_outputs)} outputs, "
            f"eager produces {len(expected_outputs)}"
        )
        return False

    all_ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_outputs, actual_outputs)):
        if not isinstance(expected_tensor, torch.Tensor) or not isinstance(actual_tensor, torch.Tensor):
            ok = expected_tensor == actual_tensor
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_ok = all_ok and bool(ok)
            continue

        metadata_ok = (
            tuple(expected_tensor.shape) == tuple(actual_tensor.shape)
            and expected_tensor.dtype == actual_tensor.dtype
            and expected_tensor.stride() == actual_tensor.stride()
            and expected_tensor.device == actual_tensor.device
        )
        if not metadata_ok:
            print(
                f"  output {index}: SCOPE_MISMATCH "
                f"oracle=(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
                f"dtype={actual_tensor.dtype} device={actual_tensor.device}) "
                f"eager=(shape={list(expected_tensor.shape)} stride={expected_tensor.stride()} "
                f"dtype={expected_tensor.dtype} device={expected_tensor.device})"
            )
            all_ok = False
            continue

        if not expected_tensor.is_floating_point():
            ok = torch.equal(expected_tensor, actual_tensor)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected_tensor.dtype})")
            all_ok = all_ok and ok
            continue

        expected_f32 = expected_tensor.float()
        actual_f32 = actual_tensor.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_ok = torch.equal(expected_nan, actual_nan)

        non_nan = ~(expected_nan | actual_nan)
        if non_nan.any():
            values_ok = torch.allclose(expected_f32[non_nan], actual_f32[non_nan], atol=atol, rtol=rtol)
        else:
            values_ok = True

        finite = torch.isfinite(expected_f32) & torch.isfinite(actual_f32)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        else:
            max_diff = 0.0

        ok = nan_ok and values_ok
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
            f"stride={expected_tensor.stride()} max_finite_diff={max_diff:.2e} "
            f"expected_nan={int(expected_nan.sum().item())} oracle_nan={int(actual_nan.sum().item())})"
        )
        all_ok = all_ok and ok

    return all_ok


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
        ok = _check_oracle_nan_aware(
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
