"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet dual batch-norm-backward return tuple by row-tiling the shared `[512, 24, 56, 56]` add/copy/slice producer, cooperatively accumulating the C=24 and sliced C=12 channel reductions in one partial/finalizer pair, and writing both returned gradient tensors with their required contiguous and channels-last layouts from one producer, whereas Inductor currently schedules the channels-last copy/contiguous clone path, slice, sibling `sum([0,2,3])` reductions, dependent BN-backward epilogues, and two tensor stores as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates a memory-format-changing producer, overlapping channel slices, dependent channel summaries, and multiple layout-distinct side outputs; the fix is COOPERATIVE_SPLIT_K: add a split-row multi-output lowering that shares compatible channel partials across overlapping slices, finalizes the small channel summaries once, and fuses all dependent BN-backward stores into the producer schedule."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_fb9996974d8f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_ghostnet_100_train_001_7c586a54"

N = 512
C24 = 24
C12 = 12
H = 56
W = 56
HW = H * W
ROWS = N * HW
INV_ROWS = 6.228077168367346e-07

REDUCE_BLOCK_ROWS = 512
STORE_BLOCK_ROWS = 64
STORE_BLOCK_C = 32
BLOCK_C24 = 32
BLOCK_C12 = 16
FINAL_BLOCK_TILES = 4096
GROUP_ROWS = 32768
GROUP_R_BLOCK = 2048
GROUP_BLOCK_C = 1
GROUP_FINAL_BLOCK_TILES = 64



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
        value.cuda()
        if isinstance(value, torch.Tensor) and value.device.type != "cuda"
        else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        clone_13,
        getitem_246,
        arg230_1,
        arg530_1,
        arg231_1,
        arg26_1,
        arg225_1,
        arg532_1,
        arg226_1,
        arg22_1,
    ) = inputs
    return (
        clone_13.contiguous(),
        getitem_246.contiguous(),
        arg230_1.contiguous(),
        arg530_1.contiguous(),
        arg231_1.contiguous(),
        arg26_1.contiguous(),
        arg225_1.contiguous(),
        arg532_1.contiguous(),
        arg226_1.contiguous(),
        arg22_1.contiguous(),
    )


@triton.jit
def _partial_reductions_kernel(
    clone_ptr,
    getitem_ptr,
    rhs24_ptr,
    mean24_ptr,
    rhs12_ptr,
    mean12_ptr,
    partial_x24_ptr,
    partial_x_rhs24_ptr,
    partial_x_rhs12_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    BLOCK_C24_: tl.constexpr,
    BLOCK_C12_: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
    c24 = tl.arange(0, BLOCK_C24_)
    c12 = tl.arange(0, BLOCK_C12_)
    row_mask = rows < ROWS_
    c24_mask = c24 < C24_
    c12_mask = c12 < C12_

    n = rows // HW_
    hw = rows - n * HW_

    offsets24 = n[None, :] * (C24_ * HW_) + c24[:, None] * HW_ + hw[None, :]
    mask24 = c24_mask[:, None] & row_mask[None, :]
    x24 = (
        tl.load(clone_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
    )
    centered24 = (
        tl.load(rhs24_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        - tl.load(mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)[:, None]
    )

    partial24_offsets = tile * C24_ + c24
    tl.store(
        partial_x24_ptr + partial24_offsets,
        tl.sum(tl.where(mask24, x24, 0.0), axis=1),
        mask=c24_mask,
    )
    tl.store(
        partial_x_rhs24_ptr + partial24_offsets,
        tl.sum(tl.where(mask24, x24 * centered24, 0.0), axis=1),
        mask=c24_mask,
    )

    x12_offsets = (
        n[None, :] * (C24_ * HW_) + (c12[:, None] + C12_) * HW_ + hw[None, :]
    )
    rhs12_offsets = n[None, :] * (C12_ * HW_) + c12[:, None] * HW_ + hw[None, :]
    mask12 = c12_mask[:, None] & row_mask[None, :]
    x12 = (
        tl.load(clone_ptr + x12_offsets, mask=mask12, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x12_offsets, mask=mask12, other=0.0).to(tl.float32)
    )
    centered12 = (
        tl.load(rhs12_ptr + rhs12_offsets, mask=mask12, other=0.0).to(tl.float32)
        - tl.load(mean12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)[:, None]
    )
    partial12_offsets = tile * C12_ + c12
    tl.store(
        partial_x_rhs12_ptr + partial12_offsets,
        tl.sum(tl.where(mask12, x12 * centered12, 0.0), axis=1),
        mask=c12_mask,
    )


@triton.jit
def _partial_grouped_reductions_kernel(
    clone_ptr,
    getitem_ptr,
    rhs24_ptr,
    mean24_ptr,
    rhs12_ptr,
    mean12_ptr,
    partial_x24_ptr,
    partial_x_rhs24_ptr,
    partial_x_rhs12_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    GROUP_ROWS_: tl.constexpr,
    R_BLOCK_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    group = tl.program_id(0)
    c24 = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    c12 = tl.where(c24 >= C12_, c24 - C12_, 0)
    c24_mask = c24 < C24_
    tail_mask = (c24 >= C12_) & (c24 < C24_)
    group_start = group * GROUP_ROWS_

    acc_x24 = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_x_rhs24 = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_x_rhs12 = tl.zeros((BLOCK_C_,), dtype=tl.float32)

    for start in tl.range(0, GROUP_ROWS_, R_BLOCK_):
        rows = group_start + start + tl.arange(0, R_BLOCK_)
        row_mask = rows < ROWS_
        n = rows // HW_
        hw = rows - n * HW_
        offsets24 = n[None, :] * (C24_ * HW_) + c24[:, None] * HW_ + hw[None, :]
        mask24 = c24_mask[:, None] & row_mask[None, :]

        x24 = (
            tl.load(clone_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        )
        centered24 = (
            tl.load(rhs24_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
            - tl.load(mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)[:, None]
        )
        acc_x24 += tl.sum(tl.where(mask24, x24, 0.0), axis=1)
        acc_x_rhs24 += tl.sum(tl.where(mask24, x24 * centered24, 0.0), axis=1)

        offsets12 = n[None, :] * (C12_ * HW_) + c12[:, None] * HW_ + hw[None, :]
        mask12 = tail_mask[:, None] & row_mask[None, :]
        centered12 = (
            tl.load(rhs12_ptr + offsets12, mask=mask12, other=0.0).to(tl.float32)
            - tl.load(mean12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)[:, None]
        )
        acc_x_rhs12 += tl.sum(tl.where(mask12, x24 * centered12, 0.0), axis=1)

    tl.store(partial_x24_ptr + group * C24_ + c24, acc_x24, mask=c24_mask)
    tl.store(partial_x_rhs24_ptr + group * C24_ + c24, acc_x_rhs24, mask=c24_mask)
    tl.store(partial_x_rhs12_ptr + group * C12_ + c12, acc_x_rhs12, mask=tail_mask)


@triton.jit
def _finalize_reductions_kernel(
    partial_x24_ptr,
    partial_x_rhs24_ptr,
    partial_x_rhs12_ptr,
    scale24_ptr,
    scale12_ptr,
    grad24_ptr,
    grad12_ptr,
    sum_x24_ptr,
    sum_x_rhs24_ptr,
    sum_x_rhs12_ptr,
    gain24_ptr,
    gain12_ptr,
    out8_ptr,
    out17_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C24_: tl.constexpr,
    BLOCK_C12_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
):
    c24 = tl.arange(0, BLOCK_C24_)
    c12 = tl.arange(0, BLOCK_C12_)
    c24_mask = c24 < C24_
    c12_mask = c12 < C12_
    tile_offsets = tl.arange(0, BLOCK_TILES_)

    acc_x24 = tl.zeros((BLOCK_C24_,), dtype=tl.float32)
    acc_x_rhs24 = tl.zeros((BLOCK_C24_,), dtype=tl.float32)
    for start in range(0, NUM_TILES_, BLOCK_TILES_):
        tiles = start + tile_offsets
        mask = (tiles[:, None] < NUM_TILES_) & c24_mask[None, :]
        offsets = tiles[:, None] * C24_ + c24[None, :]
        acc_x24 += tl.sum(
            tl.load(partial_x24_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x_rhs24 += tl.sum(
            tl.load(partial_x_rhs24_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    acc_x_rhs12 = tl.zeros((BLOCK_C12_,), dtype=tl.float32)
    for start in range(0, NUM_TILES_, BLOCK_TILES_):
        tiles = start + tile_offsets
        mask = (tiles[:, None] < NUM_TILES_) & c12_mask[None, :]
        offsets = tiles[:, None] * C12_ + c12[None, :]
        acc_x_rhs12 += tl.sum(
            tl.load(partial_x_rhs12_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    scale24 = tl.load(scale24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    scale12 = tl.load(scale12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)
    grad24 = tl.load(grad24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    grad12 = tl.load(grad12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)
    tl.store(sum_x24_ptr + c24, acc_x24 * INV_ROWS_, mask=c24_mask)
    tl.store(
        sum_x_rhs24_ptr + c24,
        acc_x_rhs24 * INV_ROWS_ * scale24 * scale24,
        mask=c24_mask,
    )
    tl.store(
        sum_x_rhs12_ptr + c12,
        acc_x_rhs12 * INV_ROWS_ * scale12 * scale12,
        mask=c12_mask,
    )
    tl.store(gain24_ptr + c24, scale24 * grad24, mask=c24_mask)
    tl.store(gain12_ptr + c12, scale12 * grad12, mask=c12_mask)
    tl.store(out8_ptr + c24, acc_x_rhs24 * scale24, mask=c24_mask)
    tl.store(out17_ptr + c12, acc_x_rhs12 * scale12, mask=c12_mask)


@triton.jit
def _store_outputs_kernel(
    clone_ptr,
    getitem_ptr,
    rhs24_ptr,
    mean24_ptr,
    scale24_ptr,
    grad24_ptr,
    rhs12_ptr,
    mean12_ptr,
    scale12_ptr,
    grad12_ptr,
    sum_x24_ptr,
    sum_x_rhs24_ptr,
    sum_x_rhs12_ptr,
    out7_ptr,
    out16_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    BLOCK_C24_: tl.constexpr,
    BLOCK_C12_: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
    c24 = tl.arange(0, BLOCK_C24_)
    c12 = tl.arange(0, BLOCK_C12_)
    row_mask = rows < ROWS_
    c24_mask = c24 < C24_
    c12_mask = c12 < C12_

    n = rows // HW_
    hw = rows - n * HW_

    offsets24 = n[None, :] * (C24_ * HW_) + c24[:, None] * HW_ + hw[None, :]
    mask24 = c24_mask[:, None] & row_mask[None, :]
    x24 = (
        tl.load(clone_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
    )
    centered24 = (
        tl.load(rhs24_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        - tl.load(mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)[:, None]
    )
    scale24 = tl.load(scale24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    grad24 = tl.load(grad24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    sum_x24 = tl.load(sum_x24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    sum_x_rhs24 = tl.load(sum_x_rhs24_ptr + c24, mask=c24_mask, other=0.0).to(
        tl.float32
    )
    correction24 = sum_x_rhs24[:, None] * INV_ROWS_ * scale24[:, None] * scale24[:, None]
    mean_grad24 = sum_x24[:, None] * INV_ROWS_
    out24 = (x24 - centered24 * correction24 - mean_grad24) * (
        scale24 * grad24
    )[:, None]
    tl.store(out7_ptr + offsets24, out24, mask=mask24)

    x12_offsets = (
        n[None, :] * (C24_ * HW_) + (c12[:, None] + C12_) * HW_ + hw[None, :]
    )
    rhs12_offsets = n[None, :] * (C12_ * HW_) + c12[:, None] * HW_ + hw[None, :]
    out16_offsets = n[None, :] * (C12_ * HW_) + hw[None, :] * C12_ + c12[:, None]
    mask12 = c12_mask[:, None] & row_mask[None, :]
    x12 = (
        tl.load(clone_ptr + x12_offsets, mask=mask12, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x12_offsets, mask=mask12, other=0.0).to(tl.float32)
    )
    centered12 = (
        tl.load(rhs12_ptr + rhs12_offsets, mask=mask12, other=0.0).to(tl.float32)
        - tl.load(mean12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)[:, None]
    )
    scale12 = tl.load(scale12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)
    grad12 = tl.load(grad12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)
    sum_x12 = tl.load(sum_x24_ptr + c12 + C12_, mask=c12_mask, other=0.0).to(
        tl.float32
    )
    sum_x_rhs12 = tl.load(sum_x_rhs12_ptr + c12, mask=c12_mask, other=0.0).to(
        tl.float32
    )
    correction12 = sum_x_rhs12[:, None] * INV_ROWS_ * scale12[:, None] * scale12[:, None]
    mean_grad12 = sum_x12[:, None] * INV_ROWS_
    out12 = (x12 - centered12 * correction12 - mean_grad12) * (
        scale12 * grad12
    )[:, None]
    tl.store(out16_ptr + out16_offsets, out12, mask=mask12)


@triton.jit
def _store_outputs_channel_block_kernel(
    clone_ptr,
    getitem_ptr,
    rhs24_ptr,
    mean24_ptr,
    rhs12_ptr,
    mean12_ptr,
    mean_x24_ptr,
    correction24_ptr,
    correction12_ptr,
    gain24_ptr,
    gain12_ptr,
    out7_ptr,
    out16_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    row_tile = tl.program_id(0)
    channel_tile = tl.program_id(1)
    rows = row_tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
    c24 = channel_tile * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    row_mask = rows < ROWS_
    c24_mask = c24 < C24_

    n = rows // HW_
    hw = rows - n * HW_

    offsets24 = n[None, :] * (C24_ * HW_) + c24[:, None] * HW_ + hw[None, :]
    mask24 = c24_mask[:, None] & row_mask[None, :]
    x24 = (
        tl.load(clone_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
    )
    centered24 = (
        tl.load(rhs24_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32)
        - tl.load(mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)[:, None]
    )
    correction24 = tl.load(correction24_ptr + c24, mask=c24_mask, other=0.0).to(
        tl.float32
    )
    mean_grad24 = tl.load(mean_x24_ptr + c24, mask=c24_mask, other=0.0).to(
        tl.float32
    )
    gain24 = tl.load(gain24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    out24 = (x24 - centered24 * correction24[:, None] - mean_grad24[:, None]) * (
        gain24
    )[:, None]
    tl.store(out7_ptr + offsets24, out24, mask=mask24)

    c12 = tl.where(c24 >= C12_, c24 - C12_, 0)
    tail_mask = (c24 >= C12_) & c24_mask
    rhs12_offsets = n[None, :] * (C12_ * HW_) + c12[:, None] * HW_ + hw[None, :]
    out16_offsets = n[None, :] * (C12_ * HW_) + hw[None, :] * C12_ + c12[:, None]
    mask12 = tail_mask[:, None] & row_mask[None, :]
    centered12 = (
        tl.load(rhs12_ptr + rhs12_offsets, mask=mask12, other=0.0).to(tl.float32)
        - tl.load(mean12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)[:, None]
    )
    correction12 = tl.load(correction12_ptr + c12, mask=tail_mask, other=0.0).to(
        tl.float32
    )
    gain12 = tl.load(gain12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)
    out12 = (x24 - centered12 * correction12[:, None] - mean_grad24[:, None]) * (
        gain12
    )[:, None]
    tl.store(out16_ptr + out16_offsets, out12, mask=mask12)


@triton.jit
def _store_out24_kernel(
    clone_ptr,
    getitem_ptr,
    rhs24_ptr,
    mean24_ptr,
    scale24_ptr,
    grad24_ptr,
    sum_x24_ptr,
    sum_x_rhs24_ptr,
    out7_ptr,
    C24_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    BLOCK_C24_: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
    c = tl.arange(0, BLOCK_C24_)
    row_mask = rows < ROWS_
    c_mask = c < C24_

    n = rows // HW_
    hw = rows - n * HW_
    offsets = n[None, :] * (C24_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mask = c_mask[:, None] & row_mask[None, :]

    x = (
        tl.load(clone_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    centered = (
        tl.load(rhs24_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )
    scale = tl.load(scale24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    grad = tl.load(grad24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    correction = sum_x_rhs[:, None] * INV_ROWS_ * scale[:, None] * scale[:, None]
    mean_grad = sum_x[:, None] * INV_ROWS_
    out = (x - centered * correction - mean_grad) * (scale * grad)[:, None]
    tl.store(out7_ptr + offsets, out, mask=mask)


@triton.jit
def _store_out12_kernel(
    clone_ptr,
    getitem_ptr,
    rhs12_ptr,
    mean12_ptr,
    scale12_ptr,
    grad12_ptr,
    sum_x24_ptr,
    sum_x_rhs12_ptr,
    out16_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    BLOCK_C12_: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
    c = tl.arange(0, BLOCK_C12_)
    row_mask = rows < ROWS_
    c_mask = c < C12_

    n = rows // HW_
    hw = rows - n * HW_
    x_offsets = n[None, :] * (C24_ * HW_) + (c[:, None] + C12_) * HW_ + hw[None, :]
    rhs_offsets = n[None, :] * (C12_ * HW_) + c[:, None] * HW_ + hw[None, :]
    out_offsets = n[None, :] * (C12_ * HW_) + hw[None, :] * C12_ + c[:, None]
    mask = c_mask[:, None] & row_mask[None, :]

    x = (
        tl.load(clone_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    )
    centered = (
        tl.load(rhs12_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )
    scale = tl.load(scale12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    grad = tl.load(grad12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x24_ptr + c + C12_, mask=c_mask, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    correction = sum_x_rhs[:, None] * INV_ROWS_ * scale[:, None] * scale[:, None]
    mean_grad = sum_x[:, None] * INV_ROWS_
    out = (x - centered * correction - mean_grad) * (scale * grad)[:, None]
    tl.store(out16_ptr + out_offsets, out, mask=mask)


def oracle_triton_prepared(
    clone_13: torch.Tensor,
    getitem_246: torch.Tensor,
    arg230_1: torch.Tensor,
    arg530_1: torch.Tensor,
    arg231_1: torch.Tensor,
    arg26_1: torch.Tensor,
    arg225_1: torch.Tensor,
    arg532_1: torch.Tensor,
    arg226_1: torch.Tensor,
    arg22_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if clone_13.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    assert clone_13.shape == (N, C24, H, W)
    assert getitem_246.shape == (N, C24, H, W)
    assert arg230_1.shape == (N, C24, H, W)
    assert arg530_1.shape == (1, C24, 1, 1)
    assert arg231_1.shape == (C24,)
    assert arg26_1.shape == (C24,)
    assert arg225_1.shape == (N, C12, H, W)
    assert arg532_1.shape == (1, C12, 1, 1)
    assert arg226_1.shape == (C12,)
    assert arg22_1.shape == (C12,)

    device = clone_13.device
    num_tiles = triton.cdiv(ROWS, GROUP_ROWS)
    partial_x24 = torch.empty((num_tiles, C24), device=device, dtype=torch.float32)
    partial_x_rhs24 = torch.empty((num_tiles, C24), device=device, dtype=torch.float32)
    partial_x_rhs12 = torch.empty((num_tiles, C12), device=device, dtype=torch.float32)

    _partial_grouped_reductions_kernel[(num_tiles, triton.cdiv(C24, GROUP_BLOCK_C))](
        clone_13,
        getitem_246,
        arg230_1,
        arg530_1,
        arg225_1,
        arg532_1,
        partial_x24,
        partial_x_rhs24,
        partial_x_rhs12,
        C24_=C24,
        C12_=C12,
        ROWS_=ROWS,
        HW_=HW,
        GROUP_ROWS_=GROUP_ROWS,
        R_BLOCK_=GROUP_R_BLOCK,
        BLOCK_C_=GROUP_BLOCK_C,
        num_warps=8,
    )

    sum_x24 = torch.empty((C24,), device=device, dtype=torch.float32)
    sum_x_rhs24 = torch.empty((C24,), device=device, dtype=torch.float32)
    sum_x_rhs12 = torch.empty((C12,), device=device, dtype=torch.float32)
    gain24 = torch.empty((C24,), device=device, dtype=torch.float32)
    gain12 = torch.empty((C12,), device=device, dtype=torch.float32)
    out8 = torch.empty((C24,), device=device, dtype=torch.float32)
    out17 = torch.empty((C12,), device=device, dtype=torch.float32)

    _finalize_reductions_kernel[(1,)](
        partial_x24,
        partial_x_rhs24,
        partial_x_rhs12,
        arg231_1,
        arg226_1,
        arg26_1,
        arg22_1,
        sum_x24,
        sum_x_rhs24,
        sum_x_rhs12,
        gain24,
        gain12,
        out8,
        out17,
        C24_=C24,
        C12_=C12,
        NUM_TILES_=num_tiles,
        BLOCK_TILES_=GROUP_FINAL_BLOCK_TILES,
        BLOCK_C24_=BLOCK_C24,
        BLOCK_C12_=BLOCK_C12,
        INV_ROWS_=INV_ROWS,
        num_warps=4,
    )

    out7 = torch.empty((N, C24, H, W), device=device, dtype=torch.float32)
    out16 = torch.empty_strided(
        (N, C12, H, W),
        (C12 * HW, 1, W * C12, C12),
        device=device,
        dtype=torch.float32,
    )

    _store_outputs_channel_block_kernel[
        (triton.cdiv(ROWS, STORE_BLOCK_ROWS), triton.cdiv(C24, STORE_BLOCK_C))
    ](
        clone_13,
        getitem_246,
        arg230_1,
        arg530_1,
        arg225_1,
        arg532_1,
        sum_x24,
        sum_x_rhs24,
        sum_x_rhs12,
        gain24,
        gain12,
        out7,
        out16,
        C24_=C24,
        C12_=C12,
        ROWS_=ROWS,
        HW_=HW,
        INV_ROWS_=INV_ROWS,
        BLOCK_ROWS_=STORE_BLOCK_ROWS,
        BLOCK_C_=STORE_BLOCK_C,
        num_warps=4,
    )

    return out7, out8, out16, out17


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = float("inf"), float("inf")
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


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

    print(
        f"oracle_full cooperative split-k GhostNet BN tuple: "
        f"{oracle_ms * 1000.0:.3f} us shape={SHAPE_LABEL}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=5e-3)
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
