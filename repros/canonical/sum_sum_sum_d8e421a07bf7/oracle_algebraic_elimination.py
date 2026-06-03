"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DenseNet batch-norm-backward tail from `Repro.forward` by reading the masked producer, centered input, residual slice, and affine vectors once, accumulating `sum(where)`, `sum(where * centered)`, `sum(centered)`, and `sum(slice)` per channel, then deriving both returned contiguous `[32]` vectors including the dependent final sum, whereas Inductor currently schedules the first two channel reductions, broadcast BN-backward arithmetic, slice add, and final channel reduction as separate generic pointwise/reduction work over large intermediates; Inductor cannot do this today because its scheduler/codegen preserves the literal dependent-reduction graph and does not prove the linear identity that collapses `sum((where - mean - centered * coef) * scale)` into sibling summaries; the fix is ALGEBRAIC_ELIMINATION: add a guarded dependent-reduction rewrite for this BN-backward form that introduces the needed `sum(centered)` and `sum(slice)` summaries and lowers the full tuple as one multi-accumulator reduction plus a small vector epilogue."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_d8e421a07bf7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
ADD_C = 48
C = 32
H = 32
W = 32
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 16
INV_COUNT = 7.62939453125e-06

BLOCK_K = 2048
FINAL_BLOCK_C = 16

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) and torch.cuda.is_available() else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    model = module.Repro().to(first_tensor.device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


@triton.jit
def _summary_reduce_kernel(
    add_ptr,
    mask_input_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_base_ptr,
    mean_ptr,
    partial_where_ptr,
    partial_where_centered_ptr,
    partial_centered_ptr,
    partial_slice_ptr,
    TOTAL_SPATIAL_: tl.constexpr,
    C_: tl.constexpr,
    ADD_C_: tl.constexpr,
    HW_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    c_offset = c * HW_ + hw
    dense_offsets = n * (C_ * HW_) + c_offset
    slice_offsets = n * (ADD_C_ * HW_) + (c + SLICE_START_) * HW_ + hw

    mask_input = tl.load(mask_input_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(where_rhs_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32)
    centered_base = tl.load(centered_base_ptr + dense_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    slice_value = tl.load(add_ptr + slice_offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, full_value, rhs)
    centered = centered_base - mean
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)
    slice_value = tl.where(active, slice_value, 0.0)

    partial_offset = tile * C_ + c
    tl.store(partial_where_ptr + partial_offset, tl.sum(where_value, axis=0))
    tl.store(partial_where_centered_ptr + partial_offset, tl.sum(where_value * centered, axis=0))
    tl.store(partial_centered_ptr + partial_offset, tl.sum(centered, axis=0))
    tl.store(partial_slice_ptr + partial_offset, tl.sum(slice_value, axis=0))


@triton.jit
def _finalize_vectors_kernel(
    partial_where_ptr,
    partial_where_centered_ptr,
    partial_centered_ptr,
    partial_slice_ptr,
    affine_scale_ptr,
    affine_rhs_ptr,
    out_grad_scale_ptr,
    out_slice_sum_ptr,
    NUM_TILES_: tl.constexpr,
    C_: tl.constexpr,
    INV_COUNT_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_where = tl.sum(tl.load(partial_where_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_where_centered = tl.sum(
        tl.load(partial_where_centered_ptr + offsets, mask=mask, other=0.0),
        axis=0,
    )
    sum_centered = tl.sum(tl.load(partial_centered_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_slice = tl.sum(tl.load(partial_slice_ptr + offsets, mask=mask, other=0.0), axis=0)

    c_mask = c < C_
    affine_scale = tl.load(affine_scale_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_rhs = tl.load(affine_rhs_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    coef = sum_where_centered * INV_COUNT_ * affine_scale * affine_scale
    post_scale = affine_scale * affine_rhs
    dependent_sum = sum_where - coef * sum_centered - sum_where

    tl.store(out_grad_scale_ptr + c, sum_where_centered * affine_scale, mask=c_mask)
    tl.store(out_slice_sum_ptr + c, sum_slice + dependent_sum * post_scale, mask=c_mask)


def oracle_full(
    add_23: torch.Tensor,
    arg109_1: torch.Tensor,
    full: torch.Tensor,
    getitem_150: torch.Tensor,
    arg107_1: torch.Tensor,
    arg317_1: torch.Tensor,
    arg108_1: torch.Tensor,
    arg2_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if add_23.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")

    assert add_23.shape == (N, ADD_C, H, W)
    assert arg109_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_150.shape == (N, C, H, W)
    assert arg107_1.shape == (N, C, H, W)
    assert arg317_1.shape == (1, C, 1, 1)
    assert arg108_1.shape == (C,)
    assert arg2_1.shape == (C,)

    add = add_23.contiguous()
    mask_input = arg109_1.contiguous()
    where_rhs = getitem_150.contiguous()
    centered_base = arg107_1.contiguous()
    mean = arg317_1.contiguous().view(C)
    affine_scale = arg108_1.contiguous()
    affine_rhs = arg2_1.contiguous()

    num_tiles = triton.cdiv(TOTAL_SPATIAL, BLOCK_K)
    device = add_23.device
    partial_where = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_where_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_slice = torch.empty((num_tiles, C), device=device, dtype=torch.float32)

    _summary_reduce_kernel[(C, num_tiles)](
        add,
        mask_input,
        full,
        where_rhs,
        centered_base,
        mean,
        partial_where,
        partial_where_centered,
        partial_centered,
        partial_slice,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        C_=C,
        ADD_C_=ADD_C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        NUM_TILES_=num_tiles,
        BLOCK_K_=BLOCK_K,
        num_warps=8,
    )

    out_grad_scale = torch.empty((C,), device=device, dtype=torch.float32)
    out_slice_sum = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_vectors_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_where,
        partial_where_centered,
        partial_centered,
        partial_slice,
        affine_scale,
        affine_rhs,
        out_grad_scale,
        out_slice_sum,
        NUM_TILES_=num_tiles,
        C_=C,
        INV_COUNT_=INV_COUNT,
        BLOCK_TILES_=triton.next_power_of_2(num_tiles),
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_grad_scale, out_slice_sum


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

    logical_read_bytes = (N * ADD_C * HW + N * C * HW * 3 + C + 2) * 4
    logical_write_bytes = C * 2 * 4
    print(f"oracle_full algebraic BN-dependent reductions: {oracle_us:.3f} us")
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=1e-1)
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
