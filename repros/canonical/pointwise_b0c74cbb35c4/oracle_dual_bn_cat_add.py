"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full GhostNet dual BN-inference affine plus virtual channel-cat plus final add scope by folding both branch normalizations into per-channel fp32 scale/shift coefficients and then writing the exact final fp32 `[N,2C,H,W]` tensor directly, whereas Inductor emits one generic fused pointwise kernel for the same scope but keeps the sqrt/reciprocal/broadcast affine chain in the full output-element loop; Inductor cannot do this today because its pointwise scheduler does not canonicalize BN-inference affine branches to channel coefficients or represent the fixed channel cat as a virtual producer before codegen; the fix is ALGEBRAIC_ELIMINATION: fold broadcasted BN affine branches to per-channel scale/shift and fuse fixed channel-cat consumers into the final pointwise writer."""
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

EPS = 1.0e-5
BLOCK_FLAT = 1024
BLOCK_NHWC = 256

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fold_dual_affine_kernel(
        mean0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        coeff_ptr,
        C0: tl.constexpr,
        C1: tl.constexpr,
        BLOCK_C: tl.constexpr,
        eps: tl.constexpr,
    ):
        c = tl.arange(0, BLOCK_C)
        mask1 = c < C1
        mean1 = tl.load(mean1_ptr + c, mask=mask1, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + c, mask=mask1, other=1.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c, mask=mask1, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c, mask=mask1, other=0.0).to(tl.float32)
        scale1 = weight1 * (1.0 / tl.sqrt(var1 + eps))
        shift1 = bias1 - mean1 * scale1
        tl.store(coeff_ptr + c, scale1, mask=mask1)
        tl.store(coeff_ptr + C1 + c, shift1, mask=mask1)

        mask0 = c < C0
        mean0 = tl.load(mean0_ptr + c, mask=mask0, other=0.0).to(tl.float32)
        var0 = tl.load(var0_ptr + c, mask=mask0, other=1.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + c, mask=mask0, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + c, mask=mask0, other=0.0).to(tl.float32)
        scale0 = weight0 * (1.0 / tl.sqrt(var0 + eps))
        shift0 = bias0 - mean0 * scale0
        tl.store(coeff_ptr + 2 * C1 + c, scale0, mask=mask0)
        tl.store(coeff_ptr + 2 * C1 + C0 + c, shift0, mask=mask0)

    @triton.jit
    def _dual_affine_cat_add_nchw_flat_kernel(
        conv0_ptr,
        residual_ptr,
        conv1_ptr,
        coeff_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C0: tl.constexpr,
        C1: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        hw_offsets = offsets % HW
        rows = offsets // HW
        c1_offsets = rows % C1
        n_offsets = rows // C1
        c0_offsets = c1_offsets - C0
        first_half = c1_offsets < C0

        conv1 = tl.load(conv1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale1 = tl.load(coeff_ptr + c1_offsets, mask=mask, other=0.0).to(tl.float32)
        shift1 = tl.load(coeff_ptr + C1 + c1_offsets, mask=mask, other=0.0).to(tl.float32)
        branch1 = conv1 * scale1 + shift1

        residual_offsets = (n_offsets * C0 + c1_offsets) * HW + hw_offsets
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask & first_half,
            other=0.0,
        ).to(tl.float32)

        conv0_offsets = (n_offsets * C0 + c0_offsets) * HW + hw_offsets
        conv0 = tl.load(
            conv0_ptr + conv0_offsets,
            mask=mask & ~first_half,
            other=0.0,
        ).to(tl.float32)
        scale0 = tl.load(coeff_ptr + 2 * C1 + c0_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
        shift0 = tl.load(coeff_ptr + 2 * C1 + C0 + c0_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
        branch0 = conv0 * scale0 + shift0

        cat_value = tl.where(first_half, residual, branch0)
        tl.store(out_ptr + offsets, cat_value + branch1, mask=mask)

    @triton.jit
    def _dual_affine_cat_add_nhwc_flat_kernel(
        conv0_ptr,
        residual_ptr,
        conv1_ptr,
        coeff_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C0: tl.constexpr,
        C1: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        c1_offsets = offsets % C1
        tmp = offsets // C1
        w_offsets = tmp % W
        tmp = tmp // W
        h_offsets = tmp % H
        n_offsets = tmp // H
        c0_offsets = c1_offsets - C0
        first_half = c1_offsets < C0

        conv1 = tl.load(conv1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale1 = tl.load(coeff_ptr + c1_offsets, mask=mask, other=0.0).to(tl.float32)
        shift1 = tl.load(coeff_ptr + C1 + c1_offsets, mask=mask, other=0.0).to(tl.float32)
        branch1 = conv1 * scale1 + shift1

        residual_offsets = ((n_offsets * H + h_offsets) * W + w_offsets) * C0 + c1_offsets
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask & first_half,
            other=0.0,
        ).to(tl.float32)

        conv0_offsets = ((n_offsets * H + h_offsets) * W + w_offsets) * C0 + c0_offsets
        conv0 = tl.load(
            conv0_ptr + conv0_offsets,
            mask=mask & ~first_half,
            other=0.0,
        ).to(tl.float32)
        scale0 = tl.load(coeff_ptr + 2 * C1 + c0_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
        shift0 = tl.load(coeff_ptr + 2 * C1 + C0 + c0_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
        branch0 = conv0 * scale0 + shift0

        cat_value = tl.where(first_half, residual, branch0)
        tl.store(out_ptr + offsets, cat_value + branch1, mask=mask)

def _require_f32_cuda_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {value.dtype}")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _is_nchw_contiguous(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous()


def _is_nhwc_contiguous(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous(memory_format=torch.channels_last)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dual_bn_cat_add.py")
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    names = (
        "arg322_1",
        "convolution_70",
        "arg323_1",
        "arg324_1",
        "arg325_1",
        "add_135",
        "arg332_1",
        "convolution_72",
        "arg333_1",
        "arg334_1",
        "arg335_1",
    )
    tensors = tuple(_require_f32_cuda_tensor(name, value) for name, value in zip(names, inputs))
    (
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        residual,
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
    ) = tensors

    if conv0.ndim != 4 or residual.ndim != 4 or conv1.ndim != 4:
        raise ValueError("activation inputs must be rank-4 NCHW tensors")
    n, c0, h, w = conv0.shape
    if tuple(residual.shape) != (n, c0, h, w):
        raise ValueError(f"add_135 shape {tuple(residual.shape)} does not match convolution_70 shape {tuple(conv0.shape)}")
    if tuple(conv1.shape) != (n, c0 * 2, h, w):
        raise ValueError(f"convolution_72 shape {tuple(conv1.shape)} must be {(n, c0 * 2, h, w)}")

    for name, tensor in (
        ("arg322_1", mean0),
        ("arg323_1", var0),
        ("arg324_1", weight0),
        ("arg325_1", bias0),
    ):
        if tuple(tensor.shape) != (c0,) or tuple(tensor.stride()) != (1,):
            raise ValueError(f"{name} must have shape/stride ({c0},)/(1,), got {tuple(tensor.shape)}/{tuple(tensor.stride())}")
    c1 = c0 * 2
    for name, tensor in (
        ("arg332_1", mean1),
        ("arg333_1", var1),
        ("arg334_1", weight1),
        ("arg335_1", bias1),
    ):
        if tuple(tensor.shape) != (c1,) or tuple(tensor.stride()) != (1,):
            raise ValueError(f"{name} must have shape/stride ({c1},)/(1,), got {tuple(tensor.shape)}/{tuple(tensor.stride())}")

    device = conv0.device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    all_nchw = all(_is_nchw_contiguous(tensor) for tensor in (conv0, residual, conv1))
    all_nhwc = all(_is_nhwc_contiguous(tensor) for tensor in (conv0, residual, conv1))
    if not (all_nchw or all_nhwc):
        raise ValueError("oracle expects captured contiguous NCHW or channels-last activation layouts")

    return tensors


def oracle_forward(inputs):
    """Run the full Repro.forward computation with the channel cat kept virtual."""
    (
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        residual,
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
    ) = _validate_inputs(inputs)

    n, c0, h, w = conv0.shape
    c1 = c0 * 2
    out = torch.empty_strided(
        tuple(conv1.shape),
        tuple(conv1.stride()),
        device=conv1.device,
        dtype=torch.float32,
    )
    coeff = torch.empty((2 * c1 + 2 * c0,), device=conv1.device, dtype=torch.float32)
    _fold_dual_affine_kernel[(1,)](
        mean0,
        var0,
        weight0,
        bias0,
        mean1,
        var1,
        weight1,
        bias1,
        coeff,
        C0=c0,
        C1=c1,
        BLOCK_C=triton.next_power_of_2(c1),
        eps=EPS,
        num_warps=1,
        num_stages=1,
    )

    if _is_nchw_contiguous(conv1):
        hw = h * w
        total = n * c1 * hw
        grid = (triton.cdiv(total, BLOCK_FLAT),)
        _dual_affine_cat_add_nchw_flat_kernel[grid](
            conv0,
            residual,
            conv1,
            coeff,
            out,
            TOTAL=total,
            C0=c0,
            C1=c1,
            HW=hw,
            BLOCK_N=BLOCK_FLAT,
            num_warps=4,
            num_stages=3,
        )
    else:
        total = n * c1 * h * w
        grid = (triton.cdiv(total, BLOCK_NHWC),)
        _dual_affine_cat_add_nhwc_flat_kernel[grid](
            conv0,
            residual,
            conv1,
            coeff,
            out,
            TOTAL=total,
            C0=c0,
            C1=c1,
            H=h,
            W=w,
            BLOCK_N=BLOCK_NHWC,
            num_warps=4,
            num_stages=3,
        )
    return out


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False

    metadata_ok = (
        eager.shape == actual.shape
        and eager.dtype == actual.dtype
        and eager.stride() == actual.stride()
        and eager.device == actual.device
    )
    if not metadata_ok:
        print(
            "  output 0: SCOPE_MISMATCH "
            f"oracle=(shape={list(actual.shape)} stride={actual.stride()} dtype={actual.dtype}) "
            f"eager=(shape={list(eager.shape)} stride={eager.stride()} dtype={eager.dtype})"
        )
        return False

    eager_f32 = eager.float()
    actual_f32 = actual.float()
    eager_nan = torch.isnan(eager_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_ok = torch.equal(eager_nan, actual_nan)
    finite = ~(eager_nan | actual_nan)
    if finite.any():
        max_diff = (eager_f32[finite] - actual_f32[finite]).abs().max().item()
        finite_ok = torch.allclose(eager_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        finite_ok = True

    ok = nan_ok and finite_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(eager.shape)} dtype={eager.dtype} stride={eager.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(eager_nan.sum().item())})"
    )
    return ok


def main():
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
        ok = _check_oracle_nan_equal(
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
