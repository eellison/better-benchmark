"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full T5 RMSNorm-backward return tuple by row-tiling the twelve-matmul dropout producer, doing each row's hidden reduction, writing the returned `[512, 8192]` input-gradient transpose view, and cooperatively accumulating the returned `[512]` weight-gradient column reduction, whereas Inductor currently schedules the long add/dropout producer, row-local sum, column `sum([0, 1])`, and final transpose as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output RMSNorm-backward template that coordinates a row-local reduction, a full transposed side output, and a sibling column accumulator from the same producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split the large row domain for compatible RMSNorm-backward column reductions, combine partial accumulators, and fuse the dependent dropout/input-gradient side store with the transpose-view return."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_fdefa5604cda"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 1024
M = BATCH * SEQ
D = 512
INV_D = 1.0 / D
DROPOUT_SCALE = 1.1111111111111112

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _rmsnorm_backward_tile_kernel(
    mm0_ptr,
    residual_ptr,
    mm1_ptr,
    mm2_ptr,
    mm3_ptr,
    mm4_ptr,
    mm5_ptr,
    mm6_ptr,
    mm7_ptr,
    mm8_ptr,
    mm9_ptr,
    mm10_ptr,
    mm11_ptr,
    dropout1_ptr,
    weight_ptr,
    x_ptr,
    rstd_ptr,
    dropout2_ptr,
    weight_grad_ptr,
    grad_md_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    INV_D_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    m_mask = m < M_
    d_mask = d < D_
    mask = m_mask[:, None] & d_mask[None, :]
    offsets = m[:, None] * D_ + d[None, :]

    dy = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy += tl.load(mm11_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    dropout1 = tl.load(dropout1_ptr + offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

    dy = dy * dropout1 * DROPOUT_SCALE_
    xhat = x * rstd[:, None]
    weighted = dy * weight[None, :]
    row_dot = tl.sum(tl.where(mask, weighted * x, 0.0), axis=1)

    rstd3 = rstd * rstd * rstd
    row_term = ((row_dot * -0.5) * rstd3) * INV_D_
    grad = weighted * rstd[:, None] + row_term[:, None] * (x * 2.0)

    dropout2 = tl.load(dropout2_ptr + offsets, mask=mask, other=0).to(tl.float32)
    grad = grad * dropout2 * DROPOUT_SCALE_
    tl.store(grad_md_ptr + offsets, grad, mask=mask)

    partial = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)
    tl.atomic_add(weight_grad_ptr + d, partial, sem="relaxed", mask=d_mask)


@triton.jit
def _finalize_weight_grad_kernel(
    partial_weight_grad_ptr,
    out_weight_grad_ptr,
    NUM_ROW_TILES: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tiles = tl.arange(0, BLOCK_TILES)
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (tiles[:, None] < NUM_ROW_TILES) & (d[None, :] < D_)
    offsets = tiles[:, None] * D_ + d[None, :]
    partials = tl.load(partial_weight_grad_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(out_weight_grad_ptr + d, tl.sum(partials, axis=0), mask=d < D_)


def oracle_full(
    mm_9: torch.Tensor,
    arg436_1: torch.Tensor,
    mm_11: torch.Tensor,
    mm_29: torch.Tensor,
    mm_31: torch.Tensor,
    mm_49: torch.Tensor,
    mm_51: torch.Tensor,
    mm_69: torch.Tensor,
    mm_71: torch.Tensor,
    mm_89: torch.Tensor,
    mm_91: torch.Tensor,
    mm_109: torch.Tensor,
    mm_111: torch.Tensor,
    arg217_1: torch.Tensor,
    arg50_1: torch.Tensor,
    arg215_1: torch.Tensor,
    arg216_1: torch.Tensor,
    arg214_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
    _shape_param_10,
    _shape_param_11,
    _shape_param_12,
    _shape_param_13,
    _shape_param_14,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
        _shape_param_14,
    )

    mm_inputs = (
        mm_9,
        mm_11,
        mm_29,
        mm_31,
        mm_49,
        mm_51,
        mm_69,
        mm_71,
        mm_89,
        mm_91,
        mm_109,
        mm_111,
    )
    for mm in mm_inputs:
        assert mm.shape == (M, D)
        assert mm.is_contiguous()
    assert arg436_1.shape == (BATCH, SEQ, D)
    assert arg217_1.shape == (BATCH, SEQ, D)
    assert arg50_1.shape == (D,)
    assert arg215_1.shape == (BATCH, SEQ, D)
    assert arg216_1.shape == (BATCH, SEQ, 1)
    assert arg214_1.shape == (BATCH, SEQ, D)
    assert arg436_1.is_contiguous()
    assert arg217_1.is_contiguous()
    assert arg50_1.is_contiguous()
    assert arg215_1.is_contiguous()
    assert arg216_1.is_contiguous()
    assert arg214_1.is_contiguous()

    block_m = 16
    block_d = 512
    num_row_tiles = triton.cdiv(M, block_m)
    weight_grad = torch.zeros((D,), device=arg436_1.device, dtype=torch.float32)
    grad_md = torch.empty((M, D), device=arg436_1.device, dtype=torch.float32)

    _rmsnorm_backward_tile_kernel[(num_row_tiles,)](
        mm_9,
        arg436_1,
        mm_11,
        mm_29,
        mm_31,
        mm_49,
        mm_51,
        mm_69,
        mm_71,
        mm_89,
        mm_91,
        mm_109,
        mm_111,
        arg217_1,
        arg50_1,
        arg215_1,
        arg216_1,
        arg214_1,
        weight_grad,
        grad_md,
        M_=M,
        D_=D,
        INV_D_=INV_D,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=8,
    )

    return weight_grad, grad_md.permute(1, 0)


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


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
        actual = _as_tuple(oracle_full(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(rep: int, warmup: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_full(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_full(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_full cooperative split-k T5 RMSNorm backward tuple: {oracle_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
