"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` Swin window-reverse layer-norm-backward/drop-path return tuple by row-tiling the `[25088, 512]` producer, reconstructing the fused window layout, writing the returned non-contiguous `[512, 25088]` transposed dropped-gradient side output, and cooperatively accumulating the two pre-drop `[512]` column reductions plus the post-drop `[512]` column reduction from the same tiles, whereas Inductor currently schedules the reshape/view/permute/clone layout change, hidden-dimension row reductions, drop-path epilogue, transposed side-output store, and sibling `sum([0, 1, 2])`/`sum([0])` reductions as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps a layout-changing producer, row-local reduction scalars, a dependent full-tensor side store, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, fuse the window-layout remap and drop-path transpose store, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_ba765a70455b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_34e07cb7"

BATCH = 128
HEIGHT = 14
WIDTH = 14
WINDOW = 7
WINDOW_BLOCKS_W = 2
HW = HEIGHT * WIDTH
ROWS = BATCH * HW
CHANNELS = 512
KEEP_PROB = 0.9782608672976494

TILE_ROWS = 8
TILE_CHANNELS = 512
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 256



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
        mm_146,
        arg50_1,
        arg253_1,
        arg529_1,
        view_575,
        arg252_1,
        *_shape_params,
    ) = inputs

    return (
        mm_146.reshape(ROWS, CHANNELS).contiguous(),
        arg50_1.contiguous(),
        arg253_1.reshape(ROWS, CHANNELS).contiguous(),
        arg529_1.reshape(ROWS).contiguous(),
        view_575.reshape(ROWS, CHANNELS).contiguous(),
        arg252_1.reshape(BATCH).contiguous(),
    )


@triton.jit
def _row_tile_store_and_reduce_kernel(
    mm_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    drop_mask_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_dropped_ptr,
    out_transposed_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    HW_: tl.constexpr,
    WIDTH_: tl.constexpr,
    WINDOW_: tl.constexpr,
    WINDOW_BLOCKS_W_: tl.constexpr,
    KEEP_PROB_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    row_mask = rows < ROWS_
    channel_mask = channels < CHANNELS_
    mask = row_mask[:, None] & channel_mask[None, :]

    spatial = rows % HW_
    h = spatial // WIDTH_
    w = spatial - h * WIDTH_
    block_h = h // WINDOW_
    inner_h = h - block_h * WINDOW_
    block_w = w // WINDOW_
    inner_w = w - block_w * WINDOW_
    source_rows = (
        (rows // HW_) * HW_
        + block_h * (WINDOW_BLOCKS_W_ * WINDOW_ * WINDOW_)
        + block_w * (WINDOW_ * WINDOW_)
        + inner_h * WINDOW_
        + inner_w
    )

    source_offsets = source_rows[:, None] * CHANNELS_ + channels[None, :]
    image_offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(mm_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + image_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + image_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    keep = tl.load(drop_mask_ptr + rows // HW_, mask=row_mask, other=0).to(tl.float32)
    keep_scale = keep / KEEP_PROB_

    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    grad = scale[:, None] * (
        weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
    )
    dropped = (residual + grad) * keep_scale[:, None]

    tl.store(out_transposed_ptr + image_offsets, dropped, mask=mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_dropped = tl.sum(tl.where(mask, dropped, 0.0), axis=0)
    tl.store(partial_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=channel_mask)
    tl.store(partial_x_ptr + partial_offsets, sum_x, mask=channel_mask)
    tl.store(partial_dropped_ptr + partial_offsets, sum_dropped, mask=channel_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_dropped_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_dropped_ptr,
    NUM_ROW_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_dropped = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_dropped += tl.sum(
            tl.load(partial_dropped_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + channels, acc_x_rhs, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_dropped_ptr + channels, acc_dropped, mask=channel_mask)


def oracle_triton_prepared(
    mm_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
    drop_mask_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mm_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_mc.shape == (ROWS, CHANNELS)
    assert weight_c.shape == (CHANNELS,)
    assert rhs_mc.shape == (ROWS, CHANNELS)
    assert scale_m.shape == (ROWS,)
    assert residual_mc.shape == (ROWS, CHANNELS)
    assert drop_mask_b.shape == (BATCH,)
    assert mm_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert scale_m.is_contiguous()
    assert residual_mc.is_contiguous()
    assert drop_mask_b.is_contiguous()

    device = mm_mc.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partial_x_rhs = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    partial_x = torch.empty((num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    partial_dropped = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    out_transposed = torch.empty_strided(
        (CHANNELS, ROWS),
        (1, CHANNELS),
        device=device,
        dtype=torch.float32,
    )

    _row_tile_store_and_reduce_kernel[(num_row_tiles,)](
        mm_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        drop_mask_b,
        partial_x_rhs,
        partial_x,
        partial_dropped,
        out_transposed,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        HW_=HW,
        WIDTH_=WIDTH,
        WINDOW_=WINDOW,
        WINDOW_BLOCKS_W_=WINDOW_BLOCKS_W,
        KEEP_PROB_=KEEP_PROB,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=8,
    )

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_dropped = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_x_rhs,
        partial_x,
        partial_dropped,
        out_x_rhs,
        out_x,
        out_dropped,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_transposed, out_dropped


def oracle_full(
    *inputs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


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
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
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

    logical_read_bytes = (
        ROWS * CHANNELS * 4
        + CHANNELS * 4
        + ROWS * CHANNELS * 4
        + ROWS * 4
        + ROWS * CHANNELS * 4
        + BATCH
    )
    logical_write_bytes = ROWS * CHANNELS * 4 + 3 * CHANNELS * 4
    partial_bytes = triton.cdiv(ROWS, TILE_ROWS) * CHANNELS * 4 * 3
    print(
        f"oracle_full cooperative split-k Swin window LN/drop-path tuple: "
        f"{oracle_ms * 1000.0:.3f} us shape={SHAPE_LABEL}"
    )
    print(
        f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB "
        f"partial traffic: {partial_bytes / 1e6:.1f} MB"
    )


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
