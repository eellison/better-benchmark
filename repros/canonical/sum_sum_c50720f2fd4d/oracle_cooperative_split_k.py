"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ShuffleNet average-pool-gradient plus ReLU-gated batch-norm-backward return tuple from Repro.forward by cooperatively split-K reducing both per-channel summaries over the shared N/H/W domain, then using the finalized summaries in one Triton epilogue that writes the full [512,1024,7,7] input-gradient tensor and [1024] scale-gradient vector, whereas Inductor currently schedules the expand/div producer, ReLU-gated where, sibling channel sums, and dependent BN-backward epilogues as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions with a dependent full-tensor side-output epilogue fed by a broadcasted average-pool producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across the reduced N/H/W dimension, combine partial summaries once, and fuse the average-pool-gradient producer plus downstream tensor/vector epilogues."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_c50720f2fd4d"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 1024
H = 7
W = 7
HW = H * W
TOTAL_REDUCTION = N * HW
NUMEL = N * C * HW
POOL_DENOM = 49.0
BN_SCALE = 3.985969387755102e-05

SUMMARY_BLOCK_C = 4
SUMMARY_BLOCK_K = 512
FINAL_BLOCK_C = 16
EPILOGUE_BLOCK = 256



@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _summary_partials_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    partial_where_ptr,
    partial_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_REDUCTION_: tl.constexpr,
    POOL_DENOM_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    k = tl.program_id(1) * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    c_mask = c < C_
    k_mask = k < TOTAL_REDUCTION_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mm_offsets = n[None, :] * C_ + c[:, None]

    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = _sub_rn_f32(x, mean[:, None])
    normalized = _mul_rn_f32(centered, invstd[:, None])
    affine = _add_rn_f32(_mul_rn_f32(normalized, weight[:, None]), bias[:, None])
    pooled_grad = _div_rn_f32(mm, POOL_DENOM_)
    where_value = tl.where(mask & (affine > 0.0), pooled_grad, 0.0)
    centered = tl.where(mask, centered, 0.0)

    partial_offset = tl.program_id(1) * C_ + c
    tl.store(
        partial_where_ptr + partial_offset,
        tl.sum(where_value, axis=1),
        mask=c_mask,
    )
    tl.store(
        partial_centered_ptr + partial_offset,
        tl.sum(_mul_rn_f32(where_value, centered), axis=1),
        mask=c_mask,
    )


@triton.jit
def _finalize_summaries_kernel(
    partial_where_ptr,
    partial_centered_ptr,
    invstd_ptr,
    sum_where_ptr,
    sum_centered_ptr,
    out_scale_ptr,
    C_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_where = tl.sum(tl.load(partial_where_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_centered = tl.sum(
        tl.load(partial_centered_ptr + offsets, mask=mask, other=0.0),
        axis=0,
    )
    c_mask = c < C_
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum_where_ptr + c, sum_where, mask=c_mask)
    tl.store(sum_centered_ptr + c, sum_centered, mask=c_mask)
    tl.store(out_scale_ptr + c, _mul_rn_f32(sum_centered, invstd), mask=c_mask)


@triton.jit
def _bn_epilogue_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    sum_where_ptr,
    sum_centered_ptr,
    out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    POOL_DENOM_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_ELEMS_: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)

    mm = tl.load(mm_ptr + n * C_ + c, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_centered = tl.load(sum_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _sub_rn_f32(x, mean)
    normalized = _mul_rn_f32(centered, invstd)
    affine = _add_rn_f32(_mul_rn_f32(normalized, weight), bias)
    pooled_grad = _div_rn_f32(mm, POOL_DENOM_)
    where_value = tl.where(affine > 0.0, pooled_grad, 0.0)

    mean_term = _mul_rn_f32(sum_where, BN_SCALE_)
    centered_sum_scaled = _mul_rn_f32(sum_centered, BN_SCALE_)
    variance_term = _mul_rn_f32(centered_sum_scaled, _mul_rn_f32(invstd, invstd))
    centered_variance = _mul_rn_f32(centered, variance_term)
    residual = _sub_rn_f32(_sub_rn_f32(where_value, centered_variance), mean_term)
    out = _mul_rn_f32(residual, _mul_rn_f32(invstd, weight))
    tl.store(out_ptr + offsets, out, mask=active)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_full(
    mm: torch.Tensor,
    arg318_1: torch.Tensor,
    arg319_1: torch.Tensor,
    arg320_1: torch.Tensor,
    arg132_1: torch.Tensor,
    arg133_1: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if mm.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")

    assert _shape_param_0 == [N, C, H, W]
    assert mm.shape == (N, C)
    assert arg318_1.shape == (N, C, H, W)
    assert arg319_1.shape == (1, C, 1, 1)
    assert arg320_1.shape == (1, C, 1, 1)
    assert arg132_1.shape == (C,)
    assert arg133_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg318_1.is_contiguous()
    assert arg319_1.is_contiguous()
    assert arg320_1.is_contiguous()
    assert arg132_1.is_contiguous()
    assert arg133_1.is_contiguous()

    num_tiles = triton.cdiv(TOTAL_REDUCTION, SUMMARY_BLOCK_K)
    device = mm.device
    partial_where = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    sum_centered = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)

    _summary_partials_kernel[(triton.cdiv(C, SUMMARY_BLOCK_C), num_tiles)](
        mm,
        arg318_1,
        arg319_1,
        arg320_1,
        arg132_1,
        arg133_1,
        partial_where,
        partial_centered,
        C_=C,
        HW_=HW,
        TOTAL_REDUCTION_=TOTAL_REDUCTION,
        POOL_DENOM_=POOL_DENOM,
        BLOCK_C_=SUMMARY_BLOCK_C,
        BLOCK_K_=SUMMARY_BLOCK_K,
        num_warps=4,
    )

    _finalize_summaries_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_where,
        partial_centered,
        arg320_1,
        sum_where,
        sum_centered,
        out_scale,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES_=triton.next_power_of_2(num_tiles),
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=4,
    )

    out_grad = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    _bn_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK),)](
        mm,
        arg318_1,
        arg319_1,
        arg320_1,
        arg132_1,
        arg133_1,
        sum_where,
        sum_centered,
        out_grad,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        POOL_DENOM_=POOL_DENOM,
        BN_SCALE_=BN_SCALE,
        BLOCK_ELEMS_=EPILOGUE_BLOCK,
        num_warps=4,
    )

    return out_grad, out_scale


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} match=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_full(*inputs)
        torch.cuda.synchronize()
        oracle_us = (
            triton.testing.do_bench(
                lambda: oracle_full(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            )
            * 1000.0
        )

    logical_read_bytes = (NUMEL * 2 + N * C + C * 4) * 4
    logical_write_bytes = (NUMEL + C) * 4
    print(f"oracle_full cooperative split-k ReLU BN backward: {oracle_us:.3f} us")
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")
    print(f"oracle_us={oracle_us:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
