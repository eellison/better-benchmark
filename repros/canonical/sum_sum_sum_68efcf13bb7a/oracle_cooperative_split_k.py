"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full BERT layer-norm-backward/dropout return tuple by rebuilding the four-input `[16384, 768]` add producer inside a row-tiled Triton kernel, reducing each token row for the dependent gradient formula, writing the returned `[768, 16384]` transpose-view side output, and cooperatively accumulating the two pre-mask `[768]` column reductions plus the post-mask `[768]` column reduction from the same tiles, whereas Inductor currently schedules the add chain, row-local hidden reductions, dropout-mask epilogue, transpose layout, and sibling `sum([0, 1])`/`sum([0])` column reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reduction scalars, a dependent full-tensor side output, a shared multi-input producer, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across token tiles, fuse shared add producers plus transposed side-output stores, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_68efcf13bb7a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_bertformaskedlm_train_001_a72a45d6"

BATCH = 32
SEQ = 512
M = BATCH * SEQ
C = 768
DROP_SCALE = 1.1111111111111112

TILE_M = 4
BLOCK_C = 1024
FINAL_BLOCK_C = 16
FINAL_BLOCK_TILES = 256

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
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_130,
        mul_317,
        mm_132,
        mm_134,
        arg11_1,
        arg120_1,
        arg317_1,
        arg119_1,
        *_shape_params,
    ) = inputs

    return (
        mm_130.reshape(M, C).contiguous(),
        mul_317.reshape(M, C).contiguous(),
        mm_132.reshape(M, C).contiguous(),
        mm_134.reshape(M, C).contiguous(),
        arg11_1.contiguous(),
        arg120_1.reshape(M, C).contiguous(),
        arg317_1.reshape(M).contiguous(),
        arg119_1.reshape(M, C).contiguous(),
    )


@triton.jit
def _row_tile_store_and_reduce_kernel(
    mm130_ptr,
    mul317_ptr,
    mm132_ptr,
    mm134_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    drop_mask_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_grad_ptr,
    out_transposed_ptr,
    M_: tl.constexpr,
    C_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    tile_m = tl.program_id(0)
    m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C_)
    m_mask = m < M_
    c_mask = c < C_
    mask = m_mask[:, None] & c_mask[None, :]
    offsets = m[:, None] * C_ + c[None, :]

    x = (
        tl.load(mul317_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm130_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    x = x + tl.load(mm132_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x + tl.load(mm134_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    drop_mask = tl.load(drop_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    grad = row_scale[:, None] * (
        weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
    )
    masked_grad = grad * drop_mask * DROP_SCALE_
    tl.store(out_transposed_ptr + offsets, masked_grad, mask=mask)

    partial_offsets = tile_m * C_ + c
    sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_masked_grad = tl.sum(tl.where(mask, masked_grad, 0.0), axis=0)
    tl.store(partial_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, sum_x, mask=c_mask)
    tl.store(partial_masked_grad_ptr + partial_offsets, sum_masked_grad, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_masked_grad_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_masked_grad_ptr,
    NUM_M_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    c_mask = c < C_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_masked_grad = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    for tile_start in range(0, NUM_M_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_M_TILES) & c_mask[None, :]
        offsets = tiles[:, None] * C_ + c[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_masked_grad += tl.sum(
            tl.load(partial_masked_grad_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_x_ptr + c, acc_x, mask=c_mask)
    tl.store(out_masked_grad_ptr + c, acc_masked_grad, mask=c_mask)


def oracle_triton_prepared(
    mm130_mc: torch.Tensor,
    mul317_mc: torch.Tensor,
    mm132_mc: torch.Tensor,
    mm134_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    row_scale_m: torch.Tensor,
    drop_mask_mc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mm130_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm130_mc.shape == (M, C)
    assert mul317_mc.shape == (M, C)
    assert mm132_mc.shape == (M, C)
    assert mm134_mc.shape == (M, C)
    assert weight_c.shape == (C,)
    assert rhs_mc.shape == (M, C)
    assert row_scale_m.shape == (M,)
    assert drop_mask_mc.shape == (M, C)
    assert mm130_mc.is_contiguous()
    assert mul317_mc.is_contiguous()
    assert mm132_mc.is_contiguous()
    assert mm134_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert row_scale_m.is_contiguous()
    assert drop_mask_mc.is_contiguous()

    device = mm130_mc.device
    num_m_tiles = triton.cdiv(M, TILE_M)
    partial_x_rhs = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_masked_grad = torch.empty(
        (num_m_tiles, C), device=device, dtype=torch.float32
    )
    out_transposed = torch.empty_strided(
        (C, M),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_tile_store_and_reduce_kernel[(num_m_tiles,)](
        mm130_mc,
        mul317_mc,
        mm132_mc,
        mm134_mc,
        weight_c,
        rhs_mc,
        row_scale_m,
        drop_mask_mc,
        partial_x_rhs,
        partial_x,
        partial_masked_grad,
        out_transposed,
        M_=M,
        C_=C,
        DROP_SCALE_=DROP_SCALE,
        BLOCK_M=TILE_M,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_masked_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_masked_grad,
        out_x_rhs,
        out_x,
        out_masked_grad,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_transposed, out_masked_grad


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


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
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        output_ok = shape_ok and value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_triton_prepared(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_ms = triton.testing.do_bench(
            lambda: oracle_triton_prepared(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )

    logical_read_bytes = (M * C * 4) * 6 + M * 4 + (M * C)
    logical_write_bytes = (M * C * 4) + (M // TILE_M) * C * 4 * 3 + C * 4 * 3
    print(
        f"oracle_full cooperative split-k BERT four-add LN/dropout tuple: "
        f"{oracle_ms * 1000.0:.3f} us shape={SHAPE_LABEL}"
    )
    print(f"logical traffic including partials: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=3e-3)
    parser.add_argument("--atol", type=float, default=5e-2)
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
