"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet inference batchnorm plus SiLU pointwise region in one channel-tiled Triton kernel that keeps NCHW-contiguous output, reuses each per-channel mean/variance/weight/bias load across a spatial tile, and avoids flattened index div/mod in the hot path, whereas Inductor currently emits a generic fused pointwise kernel for the broadcasted NCHW expression with scalar channel parameters recovered through generic indexing; Inductor cannot do this today because the scheduler/codegen does not specialize channel-broadcast pointwise fusion into an explicit channel-by-spatial stencil domain with parameter reuse and simple affine output offsets; the fix is SCHEDULER_FUSION: add a layout-aware pointwise scheduler/codegen path for NCHW channel-broadcast regions that tiles by channel and spatial coordinates instead of lowering through generic flattened pointwise indexing."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 4, "BLOCK_S": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 8, "BLOCK_S": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 16, "BLOCK_S": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_C": 8, "BLOCK_S": 128}, num_warps=8, num_stages=4),
        ],
        key=["C", "HW"],
    )
    @triton.jit
    def _batchnorm_silu_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        s_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        s_offsets = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_c = c_offsets < C
        mask_s = s_offsets < HW

        mean = tl.load(mean_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
        inv_std = tl.rsqrt(var + 1.0e-5)

        offsets = (n * C + c_offsets[:, None]) * HW + s_offsets[None, :]
        mask = mask_c[:, None] & mask_s[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * inv_std[:, None]
        y = y * weight[:, None] + bias[:, None]
        out = y * tl.sigmoid(y)
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_batchnorm_silu.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    for name, tensor in (
        ("arg287_1", mean),
        ("convolution_75", x),
        ("arg288_1", var),
        ("arg289_1", weight),
        ("arg290_1", bias),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must be float16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must use the captured contiguous layout")

    if x.ndim != 4:
        raise ValueError(f"convolution_75 must be rank 4, got shape {tuple(x.shape)}")
    n, c, h, w = x.shape
    if n != 64:
        raise ValueError(f"unexpected batch size {n}; expected captured batch size 64")
    for name, tensor in (("arg287_1", mean), ("arg288_1", var), ("arg289_1", weight), ("arg290_1", bias)):
        if tuple(tensor.shape) != (c,):
            raise ValueError(f"{name} shape must be ({c},), got {tuple(tensor.shape)}")
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid spatial shape {(h, w)}")

    return mean, x, var, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward batchnorm plus SiLU pointwise scope."""
    mean, x, var, weight, bias = _validate_inputs(inputs)
    n, c, h, w = x.shape
    hw = h * w

    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda meta: (n, triton.cdiv(c, meta["BLOCK_C"]), triton.cdiv(hw, meta["BLOCK_S"]))
    _batchnorm_silu_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        C=c,
        HW=hw,
    )
    return out


def _check_layout_and_values(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
        and actual.device == expected.device
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} dtype={actual.dtype})"
    )

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_mask_ok = torch.equal(expected_nan, actual_nan)
    finite = ~(expected_nan | actual_nan)
    if finite.any():
        max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        finite_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=1e-2, rtol=1e-2)
    else:
        max_diff = 0.0
        finite_ok = True
    values_ok = nan_mask_ok and finite_ok
    print(
        f"  output 0 nan_aware_values: {'PASS' if values_ok else 'FAIL'} "
        f"(max_finite_diff={max_diff:.2e} nan_count={int(actual_nan.sum().item())})"
    )
    return ok and values_ok


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

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_layout_and_values(instance, inputs)
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
