"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle cooperatively split-K-reduces the DenseNet batch-norm-backward channel summaries over the shared `(N, H, W)` domain and fuses the finalized reductions with the returned scale-gradient vector and last-32-channel residual-add input-gradient slice, whereas Inductor currently schedules the sibling channel reductions, masked producer, vector epilogue, and sliced residual-add epilogue as separate generic reduction and pointwise kernels with materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions with a dependent sliced side-output epilogue and several live same-slice residual inputs; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the reduced `(N, H, W)` domain, combine the partial summaries, and fuse the downstream vector and sliced tensor epilogues with the live upstream slice additions."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_9a2392d9eb48"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 320
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 288
SLICE_C = 32
SLICE_NUMEL = N * SLICE_C * HW
SCALE = 1.992984693877551e-05



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

    tl.atomic_add(sum_where_ptr + c, tl.sum(where_self, axis=1), sem="relaxed", mask=c_mask)
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(where_self * centered, axis=1),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _final_sum_and_slice_epilogue_kernel(
    mul_747_ptr,
    mul_765_ptr,
    mul_783_ptr,
    mul_801_ptr,
    mul_819_ptr,
    mul_837_ptr,
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
    out_sum_centered = tl.load(sum_where_centered_ptr + out_c, mask=out_c_mask, other=0.0)
    out_invstd = tl.load(invstd_ptr + out_c, mask=out_c_mask, other=0.0)
    tl.store(out_sum_ptr + out_c, out_sum_centered * out_invstd, mask=out_c_mask)

    linear = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < SLICE_NUMEL_

    hw = linear % HW_
    slice_c = (linear // HW_) % SLICE_C_
    n = linear // (SLICE_C_ * HW_)
    c = SLICE_START_ + slice_c

    slice_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + hw
    offsets_320 = n * (C_ * HW_) + c * HW_ + hw

    acc = tl.load(mul_747_ptr + n * (512 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_765_ptr + n * (480 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_783_ptr + n * (448 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_801_ptr + n * (416 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_819_ptr + n * (384 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_837_ptr + n * (352 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + offsets_320, mask=active, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets_320, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets_320, mask=active, other=0.0).to(tl.float32)
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
    mul_747: torch.Tensor,
    mul_765: torch.Tensor,
    mul_783: torch.Tensor,
    mul_801: torch.Tensor,
    mul_819: torch.Tensor,
    mul_837: torch.Tensor,
    arg326_1: torch.Tensor,
    full: torch.Tensor,
    getitem_279: torch.Tensor,
    arg324_1: torch.Tensor,
    arg705_1: torch.Tensor,
    arg325_1: torch.Tensor,
    arg54_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mul_747.shape == (N, 512, H, W)
    assert mul_765.shape == (N, 480, H, W)
    assert mul_783.shape == (N, 448, H, W)
    assert mul_801.shape == (N, 416, H, W)
    assert mul_819.shape == (N, 384, H, W)
    assert mul_837.shape == (N, 352, H, W)
    assert arg326_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_279.shape == (N, C, H, W)
    assert arg324_1.shape == (N, C, H, W)
    assert arg705_1.shape == (1, C, 1, 1)
    assert arg325_1.shape == (C,)
    assert arg54_1.shape == (C,)
    assert mul_747.is_contiguous()
    assert mul_765.is_contiguous()
    assert mul_783.is_contiguous()
    assert mul_801.is_contiguous()
    assert mul_819.is_contiguous()
    assert mul_837.is_contiguous()
    assert arg326_1.is_contiguous()
    assert getitem_279.is_contiguous()
    assert arg324_1.is_contiguous()
    assert arg705_1.is_contiguous()
    assert arg325_1.is_contiguous()
    assert arg54_1.is_contiguous()

    device = arg326_1.device
    mean = arg705_1.reshape(C)
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 8
    block_k = 512
    _masked_bn_dual_reduce_split_k_kernel[
        (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    ](
        arg326_1,
        full,
        getitem_279,
        arg324_1,
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
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        arg326_1,
        full,
        getitem_279,
        arg324_1,
        mean,
        arg325_1,
        arg54_1,
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
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        max_abs, max_rel = _max_diff(got, ref)
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} stride={got.stride()} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
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
