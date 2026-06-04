"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full DenseNet batch-norm-backward tail by cooperatively split-K reducing the two per-channel summaries over the shared N/H/W domain and then using those finalized summaries in one Triton epilogue that writes both the scale-gradient vector and returned last-32-channel residual-add slice, whereas Inductor currently schedules the sibling channel reductions, masked BN-backward producer, vector epilogue, and sliced residual-add epilogue as separate generic reduction and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions with a dependent sliced side-output epilogue and several live same-slice residual inputs; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across the reduced N/H/W domain, combine partial summaries, and fuse the downstream vector and slice epilogues with the upstream residual slice additions."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_02744d87feff"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 128
H = 56
W = 56
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 96
SLICE_C = 32
SLICE_NUMEL = N * SLICE_C * HW
SCALE = 4.982461734693877e-06



@triton.jit
def _masked_bn_dual_reduce_split_k_kernel(
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < C_
    k_mask = k < TOTAL_SPATIAL_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]

    mask_input = tl.load(mask_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_self = tl.where(mask, tl.where(mask_input <= 0.0, full_value, source), 0.0)
    centered = centered_source - mean[:, None]

    tl.atomic_add(
        sum_where_ptr + c,
        tl.sum(where_self, axis=1),
        sem="relaxed",
        mask=c_mask,
    )
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(where_self * centered, axis=1),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _final_sum_and_slice_epilogue_kernel(
    mul_972_ptr,
    mul_990_ptr,
    mul_1008_ptr,
    mul_1026_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_sum_ptr,
    out_slice_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    SLICE_NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    SUM_BLOCK_C: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)

    out_c = pid * SUM_BLOCK_C + tl.arange(0, SUM_BLOCK_C)
    out_c_mask = out_c < C_
    out_sum_centered = tl.load(sum_where_centered_ptr + out_c, mask=out_c_mask, other=0.0).to(tl.float32)
    out_invstd = tl.load(invstd_ptr + out_c, mask=out_c_mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + out_c, out_sum_centered * out_invstd, mask=out_c_mask)

    linear = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < SLICE_NUMEL_

    hw = linear % HW_
    slice_c = (linear // HW_) % SLICE_C_
    n = linear // (SLICE_C_ * HW_)
    c = SLICE_START_ + slice_c

    slice_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + hw
    bn_offsets = n * (C_ * HW_) + c * HW_ + hw

    acc = tl.load(mul_972_ptr + n * (256 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_990_ptr + n * (224 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_1008_ptr + n * (192 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_1026_ptr + n * (160 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum_where * SCALE_
    variance_term = sum_where_centered * SCALE_ * invstd * invstd
    grad = (where_self - centered * variance_term - mean_term) * (invstd * affine_weight)

    tl.store(out_slice_ptr + slice_offsets, acc + grad, mask=active)


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
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mul_972: torch.Tensor,
    mul_990: torch.Tensor,
    mul_1008: torch.Tensor,
    mul_1026: torch.Tensor,
    arg262_1: torch.Tensor,
    full: torch.Tensor,
    getitem_342: torch.Tensor,
    arg260_1: torch.Tensor,
    arg726_1: torch.Tensor,
    arg261_1: torch.Tensor,
    arg12_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mul_972.shape == (N, 256, H, W)
    assert mul_990.shape == (N, 224, H, W)
    assert mul_1008.shape == (N, 192, H, W)
    assert mul_1026.shape == (N, 160, H, W)
    assert arg262_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_342.shape == (N, C, H, W)
    assert arg260_1.shape == (N, C, H, W)
    assert arg726_1.shape == (1, C, 1, 1)
    assert arg261_1.shape == (C,)
    assert arg12_1.shape == (C,)
    assert mul_972.is_contiguous()
    assert mul_990.is_contiguous()
    assert mul_1008.is_contiguous()
    assert mul_1026.is_contiguous()
    assert arg262_1.is_contiguous()
    assert getitem_342.is_contiguous()
    assert arg260_1.is_contiguous()
    assert arg726_1.is_contiguous()
    assert arg261_1.is_contiguous()
    assert arg12_1.is_contiguous()

    device = arg262_1.device
    mean = arg726_1.reshape(C)
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 8
    block_k = 1024
    _masked_bn_dual_reduce_split_k_kernel[
        (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    ](
        arg262_1,
        full,
        getitem_342,
        arg260_1,
        mean,
        sum_where,
        sum_where_centered,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=4,
    )

    out_sum = torch.empty((C,), device=device, dtype=torch.float32)
    out_slice = torch.empty((N, SLICE_C, H, W), device=device, dtype=torch.float32)

    sum_block_c = 4
    block_elems = 256
    epilogue_grid = (
        max(triton.cdiv(C, sum_block_c), triton.cdiv(SLICE_NUMEL, block_elems)),
    )
    _final_sum_and_slice_epilogue_kernel[epilogue_grid](
        mul_972,
        mul_990,
        mul_1008,
        mul_1026,
        arg262_1,
        full,
        getitem_342,
        arg260_1,
        mean,
        arg261_1,
        arg12_1,
        sum_where,
        sum_where_centered,
        out_sum,
        out_slice,
        C_=C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SLICE_NUMEL_=SLICE_NUMEL,
        SCALE_=SCALE,
        SUM_BLOCK_C=sum_block_c,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out_sum, out_slice


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


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
        actual = _as_tuple(oracle_fused(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs = float("inf")
            max_rel = float("inf")
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_fused(*inputs)
        torch.cuda.synchronize()
        oracle_us = (
            triton.testing.do_bench(
                lambda: oracle_fused(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            )
            * 1000.0
        )

    print(f"oracle_fused full-scope cooperative split-k: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    ]

    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = (
                triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                )
                * 1000.0
            )
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
