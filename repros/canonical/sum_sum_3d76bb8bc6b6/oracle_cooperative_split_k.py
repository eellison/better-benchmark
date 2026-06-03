"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full DenseNet batch-norm-backward return tuple by split-K reducing both per-channel summaries over the shared `(N, H, W)` domain, then using the finalized summaries in one epilogue that writes `sum(where_self * centered) * arg114` and the returned non-contiguous first-16-channel slice after adding the live `add_22[:, 16:64]` slice, whereas Inductor currently schedules the sibling channel reductions, dependent broadcast arithmetic, and sliced add output as separate generic reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible small-output channel reductions with a dependent sliced side-output epilogue and preserves the target output layout; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN-backward channel reductions across the reduced N/H/W domain, combine partial summaries, and fuse the downstream vector and sliced tensor epilogues without materializing the full intermediate producer."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_3d76bb8bc6b6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
ADD_C = 64
C = 48
H = 32
W = 32
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START_IN_ADD = 16
OUT_SLICE_C = 16
OUT_SLICE_NUMEL = N * OUT_SLICE_C * HW
SCALE = 7.62939453125e-06

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _where_centered_dual_reduce_kernel(
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
def _bn_backward_slice_epilogue_kernel(
    add_22_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_sum_ptr,
    full_out_ptr,
    ADD_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SLICE_START_IN_ADD_: tl.constexpr,
    OUT_SLICE_C_: tl.constexpr,
    OUT_SLICE_NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    SUM_BLOCK_C: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)

    out_c = pid * SUM_BLOCK_C + tl.arange(0, SUM_BLOCK_C)
    out_c_mask = out_c < C_
    out_sum_centered = tl.load(sum_where_centered_ptr + out_c, mask=out_c_mask, other=0.0).to(tl.float32)
    out_rsqrt = tl.load(rsqrt_ptr + out_c, mask=out_c_mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + out_c, out_sum_centered * out_rsqrt, mask=out_c_mask)

    linear = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < OUT_SLICE_NUMEL_

    hw = linear % HW_
    slice_c = (linear // HW_) % OUT_SLICE_C_
    n = linear // (OUT_SLICE_C_ * HW_)
    c = slice_c
    add_c = SLICE_START_IN_ADD_ + slice_c

    producer_offsets = n * (C_ * HW_) + c * HW_ + hw
    add_offsets = n * (ADD_C_ * HW_) + add_c * HW_ + hw
    full_out_offsets = n * (C_ * HW_) + c * HW_ + hw

    add_value = tl.load(add_22_ptr + add_offsets, mask=active, other=0.0).to(tl.float32)
    mask_input = tl.load(mask_input_ptr + producer_offsets, mask=active, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + producer_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + producer_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum_where * SCALE_
    variance_term = sum_where_centered * SCALE_ * rsqrt * rsqrt
    grad = (where_self - centered * variance_term - mean_term) * (rsqrt * affine_weight)

    tl.store(full_out_ptr + full_out_offsets, add_value + grad, mask=active)


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
    add_22: torch.Tensor,
    arg115_1: torch.Tensor,
    full: torch.Tensor,
    getitem_144: torch.Tensor,
    arg113_1: torch.Tensor,
    arg315_1: torch.Tensor,
    arg114_1: torch.Tensor,
    arg6_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert add_22.shape == (N, ADD_C, H, W)
    assert arg115_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_144.shape == (N, C, H, W)
    assert arg113_1.shape == (N, C, H, W)
    assert arg315_1.shape == (1, C, 1, 1)
    assert arg114_1.shape == (C,)
    assert arg6_1.shape == (C,)
    assert add_22.is_contiguous()
    assert arg115_1.is_contiguous()
    assert getitem_144.is_contiguous()
    assert arg113_1.is_contiguous()
    assert arg315_1.is_contiguous()
    assert arg114_1.is_contiguous()
    assert arg6_1.is_contiguous()

    device = arg115_1.device
    mean = arg315_1.reshape(C)
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 8
    block_k = 1024
    reduce_grid = (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    _where_centered_dual_reduce_kernel[reduce_grid](
        arg115_1,
        full,
        getitem_144,
        arg113_1,
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
    out_slice = torch.empty_strided(
        (N, OUT_SLICE_C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.float32,
    )

    sum_block_c = 4
    block_elems = 1024
    epilogue_grid = (
        max(triton.cdiv(C, sum_block_c), triton.cdiv(OUT_SLICE_NUMEL, block_elems)),
    )
    _bn_backward_slice_epilogue_kernel[epilogue_grid](
        add_22,
        arg115_1,
        full,
        getitem_144,
        arg113_1,
        mean,
        arg114_1,
        arg6_1,
        sum_where,
        sum_where_centered,
        out_sum,
        out_slice,
        ADD_C_=ADD_C,
        C_=C,
        HW_=HW,
        SLICE_START_IN_ADD_=SLICE_START_IN_ADD,
        OUT_SLICE_C_=OUT_SLICE_C,
        OUT_SLICE_NUMEL_=OUT_SLICE_NUMEL,
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
        offset_ok = got.storage_offset() == ref.storage_offset()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs = float("inf")
            max_rel = float("inf")
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and offset_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} storage_offset={got.storage_offset()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} offset_match={offset_ok} allclose={value_ok}"
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
