"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BERT layer-norm-backward-style return tuple from Repro.forward, including the input sum, the normalized activation/gradient sum, the dropout-masked full transposed side output, and that side output's column sum, whereas Inductor currently schedules the hidden-dimension row reductions, surrounding pointwise arithmetic, transpose layout, and sibling batch/sequence column reductions as separate generic reduction and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reduction scalars, dependent full-tensor side stores, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add an Inductor template that tiles the row dimension for column reductions, computes row-local summaries once, fuses the dependent epilogue store, and finalizes all sibling column partials together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_cb22a50c0d9f"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 768
INV_CHANNELS = 1.0 / CHANNELS
EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112

TILE_ROWS = 4
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 8
FINAL_BLOCK_TILES = 4096



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
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_6,
        arg95_1,
        arg277_1,
        arg276_1,
        mul_3,
        arg275_1,
        *_shape_params,
    ) = inputs

    return (
        mm_6.reshape(ROWS, CHANNELS).contiguous(),
        arg95_1.contiguous(),
        arg277_1.reshape(ROWS, CHANNELS).contiguous(),
        arg276_1.reshape(ROWS).contiguous(),
        mul_3.reshape(ROWS, CHANNELS).contiguous(),
        arg275_1.reshape(ROWS, CHANNELS).contiguous(),
    )


@triton.jit
def _row_tile_kernel(
    x_ptr,
    gamma_ptr,
    dy_ptr,
    denom_base_ptr,
    residual_ptr,
    keep_mask_ptr,
    partial_sum_x_ptr,
    partial_sum_x_dy_over_denom_ptr,
    partial_sum_out_ptr,
    out_md_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
    EPS_: tl.constexpr,
    ROW_BACKWARD_SCALE_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    row_mask = rows < ROWS_
    channel_mask = channels < CHANNELS_
    mask = row_mask[:, None] & channel_mask[None, :]
    offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    gamma = tl.load(gamma_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    denom_base = tl.load(denom_base_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    denom = denom_base + EPS_

    x_over_denom = x / denom[:, None]
    x_gamma_over_denom = x_over_denom * gamma[None, :]
    row_neg_x_gamma_sum = -tl.sum(
        tl.where(mask, x_gamma_over_denom, 0.0),
        axis=1,
    )
    row_neg_x_gamma_dy_denom2_sum = -tl.sum(
        tl.where(mask, x_gamma_over_denom * dy / denom[:, None], 0.0),
        axis=1,
    )
    row_coef = tl.where(
        denom_base == 0.0,
        0.0,
        row_neg_x_gamma_dy_denom2_sum / (denom_base * 2.0),
    ) * ROW_BACKWARD_SCALE_

    out = (
        residual
        + x_gamma_over_denom
        + row_coef[:, None] * dy
        + row_neg_x_gamma_sum[:, None] * INV_CHANNELS_
    ) * keep * DROPOUT_SCALE_
    tl.store(out_md_ptr + offsets, out, mask=mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_x_dy_over_denom = tl.sum(tl.where(mask, x_over_denom * dy, 0.0), axis=0)
    sum_out = tl.sum(tl.where(mask, out, 0.0), axis=0)
    tl.store(partial_sum_x_ptr + partial_offsets, sum_x, mask=channel_mask)
    tl.store(
        partial_sum_x_dy_over_denom_ptr + partial_offsets,
        sum_x_dy_over_denom,
        mask=channel_mask,
    )
    tl.store(partial_sum_out_ptr + partial_offsets, sum_out, mask=channel_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_x_ptr,
    partial_sum_x_dy_over_denom_ptr,
    partial_sum_out_ptr,
    out_sum_x_ptr,
    out_sum_x_dy_over_denom_ptr,
    out_sum_out_ptr,
    NUM_ROW_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_sum_x = tl.zeros((BLOCK_CHANNELS,), tl.float32)
    acc_sum_x_dy_over_denom = tl.zeros((BLOCK_CHANNELS,), tl.float32)
    acc_sum_out = tl.zeros((BLOCK_CHANNELS,), tl.float32)
    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_sum_x += tl.sum(
            tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0),
            axis=0,
        )
        acc_sum_x_dy_over_denom += tl.sum(
            tl.load(partial_sum_x_dy_over_denom_ptr + offsets, mask=mask, other=0.0),
            axis=0,
        )
        acc_sum_out += tl.sum(
            tl.load(partial_sum_out_ptr + offsets, mask=mask, other=0.0),
            axis=0,
        )

    tl.store(out_sum_x_ptr + channels, acc_sum_x, mask=channel_mask)
    tl.store(
        out_sum_x_dy_over_denom_ptr + channels,
        acc_sum_x_dy_over_denom,
        mask=channel_mask,
    )
    tl.store(out_sum_out_ptr + channels, acc_sum_out, mask=channel_mask)


def oracle_full_prepared(
    x_md: torch.Tensor,
    gamma_d: torch.Tensor,
    dy_md: torch.Tensor,
    denom_base_m: torch.Tensor,
    residual_md: torch.Tensor,
    keep_mask_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")

    assert x_md.shape == (ROWS, CHANNELS)
    assert gamma_d.shape == (CHANNELS,)
    assert dy_md.shape == (ROWS, CHANNELS)
    assert denom_base_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert keep_mask_md.shape == (ROWS, CHANNELS)
    assert x_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert dy_md.is_contiguous()
    assert denom_base_m.is_contiguous()
    assert residual_md.is_contiguous()
    assert keep_mask_md.is_contiguous()

    device = x_md.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partial_sum_x = torch.empty((num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    partial_sum_x_dy_over_denom = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    partial_sum_out = torch.empty((num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    out_md = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_row_tiles,)](
        x_md,
        gamma_d,
        dy_md,
        denom_base_m,
        residual_md,
        keep_mask_md,
        partial_sum_x,
        partial_sum_x_dy_over_denom,
        partial_sum_out,
        out_md,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        EPS_=EPS,
        ROW_BACKWARD_SCALE_=ROW_BACKWARD_SCALE,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=4,
    )

    sum_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    sum_x_dy_over_denom = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_sum_x,
        partial_sum_x_dy_over_denom,
        partial_sum_out,
        sum_x,
        sum_x_dy_over_denom,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return sum_x, sum_x_dy_over_denom, out_md.t(), sum_out


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_full_prepared(*prepare_oracle_inputs(*inputs))


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
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
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
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
        oracle_full_prepared(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_ms = triton.testing.do_bench(
            lambda: oracle_full_prepared(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )

    logical_read_bytes = (
        ROWS * CHANNELS * 4 * 3
        + ROWS * CHANNELS
        + CHANNELS * 4
        + ROWS * 4
    )
    logical_write_bytes = ROWS * CHANNELS * 4 + CHANNELS * 4 * 3
    print(
        f"oracle_full cooperative split-k BERT layernorm tail: {oracle_ms * 1000.0:.3f} us "
        f"(warmup={warmup}, rep={rep})"
    )
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1.0e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the full Repro.forward tuple")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    parser.add_argument("--atol", type=float, default=5.0e-2)
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
