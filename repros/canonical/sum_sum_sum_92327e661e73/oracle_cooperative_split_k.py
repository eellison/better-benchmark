"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete channels-last Adv-Inception `mm / 64` fanout plus all six ReLU-gated batch-norm-backward branch reductions and dependent tensor/vector epilogues with one coalesced split-K Triton reduction followed by one channels-last epilogue, whereas Inductor currently schedules the expanded `mm`, sliced views, ReLU masks, twelve `sum([0, 2, 3])` reductions, and BN-backward epilogues as separate pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative multi-output reduction template that coordinates disjoint channels-last channel slices with finalized per-channel summaries feeding full-tensor side-output epilogues; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible `(N, H, W)` channel reductions, reuse the shared sliced `mm / 64` producer, and fuse each branch's gradient/vector epilogue while preserving channels-last layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
MM_C = 2048
H = 8
W = 8
HW = H * W
TOTAL_SPATIAL = N * HW
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL


def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _all_branches_channels_last_kernel(
    mm_ptr,
    x320_ptr,
    mean320_ptr,
    rsqrt320_ptr,
    weight320_ptr,
    bias320_ptr,
    x384_a_ptr,
    mean384_a_ptr,
    rsqrt384_a_ptr,
    weight384_a_ptr,
    bias384_a_ptr,
    x384_b_ptr,
    mean384_b_ptr,
    rsqrt384_b_ptr,
    weight384_b_ptr,
    bias384_b_ptr,
    x384_c_ptr,
    mean384_c_ptr,
    rsqrt384_c_ptr,
    weight384_c_ptr,
    bias384_c_ptr,
    x384_d_ptr,
    mean384_d_ptr,
    rsqrt384_d_ptr,
    weight384_d_ptr,
    bias384_d_ptr,
    x192_ptr,
    mean192_ptr,
    rsqrt192_ptr,
    weight192_ptr,
    bias192_ptr,
    grad320_ptr,
    weight_grad320_ptr,
    grad384_a_ptr,
    weight_grad384_a_ptr,
    grad384_b_ptr,
    weight_grad384_b_ptr,
    grad384_c_ptr,
    weight_grad384_c_ptr,
    grad384_d_ptr,
    weight_grad384_d_ptr,
    grad192_ptr,
    weight_grad192_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    k_mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    mm = tl.load(mm_ptr + n * MM_C_ + c, mask=k_mask, other=0.0).to(tl.float32)

    b320 = c < 320
    b384_a = (c >= 320) & (c < 704)
    b384_b = (c >= 704) & (c < 1088)
    b384_c = (c >= 1088) & (c < 1472)
    b384_d = (c >= 1472) & (c < 1856)
    b192 = c >= 1856

    local_c = tl.where(
        b320,
        c,
        tl.where(
            b384_a,
            c - 320,
            tl.where(
                b384_b,
                c - 704,
                tl.where(b384_c, c - 1088, tl.where(b384_d, c - 1472, c - 1856)),
            ),
        ),
    )

    x = tl.full((BLOCK_K,), 0.0, tl.float32)
    mean = tl.full((), 0.0, tl.float32)
    rsqrt = tl.full((), 0.0, tl.float32)
    affine_weight = tl.full((), 0.0, tl.float32)
    affine_bias = tl.full((), 0.0, tl.float32)

    offsets320 = n * (320 * HW_) + h * (320 * W_) + w * 320 + local_c
    x += tl.load(x320_ptr + offsets320, mask=k_mask & b320, other=0.0).to(tl.float32)
    mean += tl.load(mean320_ptr + local_c, mask=b320, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt320_ptr + local_c, mask=b320, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight320_ptr + local_c, mask=b320, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias320_ptr + local_c, mask=b320, other=0.0).to(tl.float32)

    offsets384_a = n * (384 * HW_) + h * (384 * W_) + w * 384 + local_c
    x += tl.load(x384_a_ptr + offsets384_a, mask=k_mask & b384_a, other=0.0).to(tl.float32)
    mean += tl.load(mean384_a_ptr + local_c, mask=b384_a, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_a_ptr + local_c, mask=b384_a, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_a_ptr + local_c, mask=b384_a, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_a_ptr + local_c, mask=b384_a, other=0.0).to(tl.float32)

    offsets384_b = n * (384 * HW_) + h * (384 * W_) + w * 384 + local_c
    x += tl.load(x384_b_ptr + offsets384_b, mask=k_mask & b384_b, other=0.0).to(tl.float32)
    mean += tl.load(mean384_b_ptr + local_c, mask=b384_b, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_b_ptr + local_c, mask=b384_b, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_b_ptr + local_c, mask=b384_b, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_b_ptr + local_c, mask=b384_b, other=0.0).to(tl.float32)

    offsets384_c = n * (384 * HW_) + h * (384 * W_) + w * 384 + local_c
    x += tl.load(x384_c_ptr + offsets384_c, mask=k_mask & b384_c, other=0.0).to(tl.float32)
    mean += tl.load(mean384_c_ptr + local_c, mask=b384_c, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_c_ptr + local_c, mask=b384_c, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_c_ptr + local_c, mask=b384_c, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_c_ptr + local_c, mask=b384_c, other=0.0).to(tl.float32)

    offsets384_d = n * (384 * HW_) + h * (384 * W_) + w * 384 + local_c
    x += tl.load(x384_d_ptr + offsets384_d, mask=k_mask & b384_d, other=0.0).to(tl.float32)
    mean += tl.load(mean384_d_ptr + local_c, mask=b384_d, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_d_ptr + local_c, mask=b384_d, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_d_ptr + local_c, mask=b384_d, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_d_ptr + local_c, mask=b384_d, other=0.0).to(tl.float32)

    offsets192 = n * (192 * HW_) + h * (192 * W_) + w * 192 + local_c
    x += tl.load(x192_ptr + offsets192, mask=k_mask & b192, other=0.0).to(tl.float32)
    mean += tl.load(mean192_ptr + local_c, mask=b192, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt192_ptr + local_c, mask=b192, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight192_ptr + local_c, mask=b192, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias192_ptr + local_c, mask=b192, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * affine_weight + affine_bias
    masked = tl.where(k_mask & (affine > 0.0), mm * INV_HW_, 0.0)
    sum_masked = tl.sum(masked, axis=0)
    sum_masked_centered = tl.sum(masked * centered, axis=0)

    variance_term = sum_masked_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_masked * SCALE_
    grad = (masked - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    weight_grad = sum_masked_centered * rsqrt

    tl.store(grad320_ptr + offsets320, grad, mask=k_mask & b320)
    tl.store(weight_grad320_ptr + local_c, weight_grad, mask=b320)
    tl.store(grad384_a_ptr + offsets384_a, grad, mask=k_mask & b384_a)
    tl.store(weight_grad384_a_ptr + local_c, weight_grad, mask=b384_a)
    tl.store(grad384_b_ptr + offsets384_b, grad, mask=k_mask & b384_b)
    tl.store(weight_grad384_b_ptr + local_c, weight_grad, mask=b384_b)
    tl.store(grad384_c_ptr + offsets384_c, grad, mask=k_mask & b384_c)
    tl.store(weight_grad384_c_ptr + local_c, weight_grad, mask=b384_c)
    tl.store(grad384_d_ptr + offsets384_d, grad, mask=k_mask & b384_d)
    tl.store(weight_grad384_d_ptr + local_c, weight_grad, mask=b384_d)
    tl.store(grad192_ptr + offsets192, grad, mask=k_mask & b192)
    tl.store(weight_grad192_ptr + local_c, weight_grad, mask=b192)


@triton.jit
def _all_branches_reduce_split_k_channels_last_kernel(
    mm_ptr,
    x320_ptr,
    mean320_ptr,
    rsqrt320_ptr,
    weight320_ptr,
    bias320_ptr,
    x384_a_ptr,
    mean384_a_ptr,
    rsqrt384_a_ptr,
    weight384_a_ptr,
    bias384_a_ptr,
    x384_b_ptr,
    mean384_b_ptr,
    rsqrt384_b_ptr,
    weight384_b_ptr,
    bias384_b_ptr,
    x384_c_ptr,
    mean384_c_ptr,
    rsqrt384_c_ptr,
    weight384_c_ptr,
    bias384_c_ptr,
    x384_d_ptr,
    mean384_d_ptr,
    rsqrt384_d_ptr,
    weight384_d_ptr,
    bias384_d_ptr,
    x192_ptr,
    mean192_ptr,
    rsqrt192_ptr,
    weight192_ptr,
    bias192_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < MM_C_
    k_mask = k < TOTAL_SPATIAL_
    mask = k_mask[:, None] & c_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    mm = tl.load(mm_ptr + n[:, None] * MM_C_ + c[None, :], mask=mask, other=0.0).to(tl.float32)

    b320 = c < 320
    b384_a = (c >= 320) & (c < 704)
    b384_b = (c >= 704) & (c < 1088)
    b384_c = (c >= 1088) & (c < 1472)
    b384_d = (c >= 1472) & (c < 1856)
    b192 = c >= 1856
    local_c = tl.where(
        b320,
        c,
        tl.where(
            b384_a,
            c - 320,
            tl.where(
                b384_b,
                c - 704,
                tl.where(b384_c, c - 1088, tl.where(b384_d, c - 1472, c - 1856)),
            ),
        ),
    )

    x = tl.full((BLOCK_K, BLOCK_C), 0.0, tl.float32)
    mean = tl.full((BLOCK_C,), 0.0, tl.float32)
    rsqrt = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_weight = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_bias = tl.full((BLOCK_C,), 0.0, tl.float32)

    offsets320 = n[:, None] * (320 * HW_) + h[:, None] * (320 * W_) + w[:, None] * 320 + local_c[None, :]
    mask320 = mask & b320[None, :]
    x += tl.load(x320_ptr + offsets320, mask=mask320, other=0.0).to(tl.float32)
    mean += tl.load(mean320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)

    offsets384_a = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_a = mask & b384_a[None, :]
    x += tl.load(x384_a_ptr + offsets384_a, mask=mask384_a, other=0.0).to(tl.float32)
    mean += tl.load(mean384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)

    offsets384_b = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_b = mask & b384_b[None, :]
    x += tl.load(x384_b_ptr + offsets384_b, mask=mask384_b, other=0.0).to(tl.float32)
    mean += tl.load(mean384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)

    offsets384_c = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_c = mask & b384_c[None, :]
    x += tl.load(x384_c_ptr + offsets384_c, mask=mask384_c, other=0.0).to(tl.float32)
    mean += tl.load(mean384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)

    offsets384_d = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_d = mask & b384_d[None, :]
    x += tl.load(x384_d_ptr + offsets384_d, mask=mask384_d, other=0.0).to(tl.float32)
    mean += tl.load(mean384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)

    offsets192 = n[:, None] * (192 * HW_) + h[:, None] * (192 * W_) + w[:, None] * 192 + local_c[None, :]
    mask192 = mask & b192[None, :]
    x += tl.load(x192_ptr + offsets192, mask=mask192, other=0.0).to(tl.float32)
    mean += tl.load(mean192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    affine = centered * rsqrt[None, :] * affine_weight[None, :] + affine_bias[None, :]
    masked = tl.where(mask & (affine > 0.0), mm * INV_HW_, 0.0)
    tl.atomic_add(sum_masked_ptr + c, tl.sum(masked, axis=0), sem="relaxed", mask=c_mask)
    tl.atomic_add(
        sum_masked_centered_ptr + c,
        tl.sum(masked * centered, axis=0),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _all_branches_epilogue_channels_last_kernel(
    mm_ptr,
    x320_ptr,
    mean320_ptr,
    rsqrt320_ptr,
    weight320_ptr,
    bias320_ptr,
    x384_a_ptr,
    mean384_a_ptr,
    rsqrt384_a_ptr,
    weight384_a_ptr,
    bias384_a_ptr,
    x384_b_ptr,
    mean384_b_ptr,
    rsqrt384_b_ptr,
    weight384_b_ptr,
    bias384_b_ptr,
    x384_c_ptr,
    mean384_c_ptr,
    rsqrt384_c_ptr,
    weight384_c_ptr,
    bias384_c_ptr,
    x384_d_ptr,
    mean384_d_ptr,
    rsqrt384_d_ptr,
    weight384_d_ptr,
    bias384_d_ptr,
    x192_ptr,
    mean192_ptr,
    rsqrt192_ptr,
    weight192_ptr,
    bias192_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    grad320_ptr,
    weight_grad320_ptr,
    grad384_a_ptr,
    weight_grad384_a_ptr,
    grad384_b_ptr,
    weight_grad384_b_ptr,
    grad384_c_ptr,
    weight_grad384_c_ptr,
    grad384_d_ptr,
    weight_grad384_d_ptr,
    grad192_ptr,
    weight_grad192_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < MM_C_
    k_mask = k < TOTAL_SPATIAL_
    mask = k_mask[:, None] & c_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    mm = tl.load(mm_ptr + n[:, None] * MM_C_ + c[None, :], mask=mask, other=0.0).to(tl.float32)

    b320 = c < 320
    b384_a = (c >= 320) & (c < 704)
    b384_b = (c >= 704) & (c < 1088)
    b384_c = (c >= 1088) & (c < 1472)
    b384_d = (c >= 1472) & (c < 1856)
    b192 = c >= 1856
    local_c = tl.where(
        b320,
        c,
        tl.where(
            b384_a,
            c - 320,
            tl.where(
                b384_b,
                c - 704,
                tl.where(b384_c, c - 1088, tl.where(b384_d, c - 1472, c - 1856)),
            ),
        ),
    )

    x = tl.full((BLOCK_K, BLOCK_C), 0.0, tl.float32)
    mean = tl.full((BLOCK_C,), 0.0, tl.float32)
    rsqrt = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_weight = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_bias = tl.full((BLOCK_C,), 0.0, tl.float32)

    offsets320 = n[:, None] * (320 * HW_) + h[:, None] * (320 * W_) + w[:, None] * 320 + local_c[None, :]
    mask320 = mask & b320[None, :]
    x += tl.load(x320_ptr + offsets320, mask=mask320, other=0.0).to(tl.float32)
    mean += tl.load(mean320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias320_ptr + local_c, mask=c_mask & b320, other=0.0).to(tl.float32)

    offsets384_a = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_a = mask & b384_a[None, :]
    x += tl.load(x384_a_ptr + offsets384_a, mask=mask384_a, other=0.0).to(tl.float32)
    mean += tl.load(mean384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_a_ptr + local_c, mask=c_mask & b384_a, other=0.0).to(tl.float32)

    offsets384_b = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_b = mask & b384_b[None, :]
    x += tl.load(x384_b_ptr + offsets384_b, mask=mask384_b, other=0.0).to(tl.float32)
    mean += tl.load(mean384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_b_ptr + local_c, mask=c_mask & b384_b, other=0.0).to(tl.float32)

    offsets384_c = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_c = mask & b384_c[None, :]
    x += tl.load(x384_c_ptr + offsets384_c, mask=mask384_c, other=0.0).to(tl.float32)
    mean += tl.load(mean384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_c_ptr + local_c, mask=c_mask & b384_c, other=0.0).to(tl.float32)

    offsets384_d = n[:, None] * (384 * HW_) + h[:, None] * (384 * W_) + w[:, None] * 384 + local_c[None, :]
    mask384_d = mask & b384_d[None, :]
    x += tl.load(x384_d_ptr + offsets384_d, mask=mask384_d, other=0.0).to(tl.float32)
    mean += tl.load(mean384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias384_d_ptr + local_c, mask=c_mask & b384_d, other=0.0).to(tl.float32)

    offsets192 = n[:, None] * (192 * HW_) + h[:, None] * (192 * W_) + w[:, None] * 192 + local_c[None, :]
    mask192 = mask & b192[None, :]
    x += tl.load(x192_ptr + offsets192, mask=mask192, other=0.0).to(tl.float32)
    mean += tl.load(mean192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias192_ptr + local_c, mask=c_mask & b192, other=0.0).to(tl.float32)

    sum_masked = tl.load(sum_masked_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_masked_centered = tl.load(sum_masked_centered_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    centered = x - mean[None, :]
    affine = centered * rsqrt[None, :] * affine_weight[None, :] + affine_bias[None, :]
    masked = tl.where(mask & (affine > 0.0), mm * INV_HW_, 0.0)
    variance_term = sum_masked_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_masked * SCALE_
    grad = (masked - centered * variance_term[None, :] - mean_term[None, :]) * (
        rsqrt[None, :] * affine_weight[None, :]
    )
    weight_grad = sum_masked_centered * rsqrt
    first_k_block = tl.program_id(1) == 0

    tl.store(grad320_ptr + offsets320, grad, mask=mask320)
    tl.store(weight_grad320_ptr + local_c, weight_grad, mask=(c_mask & b320) & first_k_block)
    tl.store(grad384_a_ptr + offsets384_a, grad, mask=mask384_a)
    tl.store(weight_grad384_a_ptr + local_c, weight_grad, mask=(c_mask & b384_a) & first_k_block)
    tl.store(grad384_b_ptr + offsets384_b, grad, mask=mask384_b)
    tl.store(weight_grad384_b_ptr + local_c, weight_grad, mask=(c_mask & b384_b) & first_k_block)
    tl.store(grad384_c_ptr + offsets384_c, grad, mask=mask384_c)
    tl.store(weight_grad384_c_ptr + local_c, weight_grad, mask=(c_mask & b384_c) & first_k_block)
    tl.store(grad384_d_ptr + offsets384_d, grad, mask=mask384_d)
    tl.store(weight_grad384_d_ptr + local_c, weight_grad, mask=(c_mask & b384_d) & first_k_block)
    tl.store(grad192_ptr + offsets192, grad, mask=mask192)
    tl.store(weight_grad192_ptr + local_c, weight_grad, mask=(c_mask & b192) & first_k_block)


def _empty_same_layout(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)


@triton.jit
def _load_branch_channels_last(
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    local_c,
    n,
    h,
    w,
    mask,
    c_mask,
    CBRANCH: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
):
    offsets = n[:, None] * (CBRANCH * HW_) + h[:, None] * (CBRANCH * W_) + w[:, None] * CBRANCH + local_c[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + local_c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + local_c, mask=c_mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(weight_ptr + local_c, mask=c_mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(bias_ptr + local_c, mask=c_mask, other=0.0).to(tl.float32)
    return offsets, x, mean, rsqrt, affine_weight, affine_bias


@triton.jit
def _all_branches_reduce_split_k_dispatched_kernel(
    mm_ptr,
    x320_ptr,
    mean320_ptr,
    rsqrt320_ptr,
    weight320_ptr,
    bias320_ptr,
    x384_a_ptr,
    mean384_a_ptr,
    rsqrt384_a_ptr,
    weight384_a_ptr,
    bias384_a_ptr,
    x384_b_ptr,
    mean384_b_ptr,
    rsqrt384_b_ptr,
    weight384_b_ptr,
    bias384_b_ptr,
    x384_c_ptr,
    mean384_c_ptr,
    rsqrt384_c_ptr,
    weight384_c_ptr,
    bias384_c_ptr,
    x384_d_ptr,
    mean384_d_ptr,
    rsqrt384_d_ptr,
    weight384_d_ptr,
    bias384_d_ptr,
    x192_ptr,
    mean192_ptr,
    rsqrt192_ptr,
    weight192_ptr,
    bias192_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < MM_C_
    k_mask = k < TOTAL_SPATIAL_
    mask = k_mask[:, None] & c_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    mm = tl.load(mm_ptr + n[:, None] * MM_C_ + c[None, :], mask=mask, other=0.0).to(tl.float32)

    if pid_c < 20:
        local_c = c
        _offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x320_ptr, mean320_ptr, rsqrt320_ptr, weight320_ptr, bias320_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=320, HW_=HW_, W_=W_,
        )
    elif pid_c < 44:
        local_c = c - 320
        _offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_a_ptr, mean384_a_ptr, rsqrt384_a_ptr, weight384_a_ptr, bias384_a_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    elif pid_c < 68:
        local_c = c - 704
        _offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_b_ptr, mean384_b_ptr, rsqrt384_b_ptr, weight384_b_ptr, bias384_b_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    elif pid_c < 92:
        local_c = c - 1088
        _offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_c_ptr, mean384_c_ptr, rsqrt384_c_ptr, weight384_c_ptr, bias384_c_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    elif pid_c < 116:
        local_c = c - 1472
        _offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_d_ptr, mean384_d_ptr, rsqrt384_d_ptr, weight384_d_ptr, bias384_d_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    else:
        local_c = c - 1856
        _offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x192_ptr, mean192_ptr, rsqrt192_ptr, weight192_ptr, bias192_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=192, HW_=HW_, W_=W_,
        )

    centered = x - mean[None, :]
    affine = centered * rsqrt[None, :] * affine_weight[None, :] + affine_bias[None, :]
    masked = tl.where(mask & (affine > 0.0), mm * INV_HW_, 0.0)
    tl.atomic_add(sum_masked_ptr + c, tl.sum(masked, axis=0), sem="relaxed", mask=c_mask)
    tl.atomic_add(
        sum_masked_centered_ptr + c,
        tl.sum(masked * centered, axis=0),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _all_branches_epilogue_dispatched_kernel(
    mm_ptr,
    x320_ptr,
    mean320_ptr,
    rsqrt320_ptr,
    weight320_ptr,
    bias320_ptr,
    x384_a_ptr,
    mean384_a_ptr,
    rsqrt384_a_ptr,
    weight384_a_ptr,
    bias384_a_ptr,
    x384_b_ptr,
    mean384_b_ptr,
    rsqrt384_b_ptr,
    weight384_b_ptr,
    bias384_b_ptr,
    x384_c_ptr,
    mean384_c_ptr,
    rsqrt384_c_ptr,
    weight384_c_ptr,
    bias384_c_ptr,
    x384_d_ptr,
    mean384_d_ptr,
    rsqrt384_d_ptr,
    weight384_d_ptr,
    bias384_d_ptr,
    x192_ptr,
    mean192_ptr,
    rsqrt192_ptr,
    weight192_ptr,
    bias192_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    grad320_ptr,
    weight_grad320_ptr,
    grad384_a_ptr,
    weight_grad384_a_ptr,
    grad384_b_ptr,
    weight_grad384_b_ptr,
    grad384_c_ptr,
    weight_grad384_c_ptr,
    grad384_d_ptr,
    weight_grad384_d_ptr,
    grad192_ptr,
    weight_grad192_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < MM_C_
    k_mask = k < TOTAL_SPATIAL_
    mask = k_mask[:, None] & c_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    mm = tl.load(mm_ptr + n[:, None] * MM_C_ + c[None, :], mask=mask, other=0.0).to(tl.float32)

    if pid_c < 20:
        local_c = c
        offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x320_ptr, mean320_ptr, rsqrt320_ptr, weight320_ptr, bias320_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=320, HW_=HW_, W_=W_,
        )
    elif pid_c < 44:
        local_c = c - 320
        offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_a_ptr, mean384_a_ptr, rsqrt384_a_ptr, weight384_a_ptr, bias384_a_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    elif pid_c < 68:
        local_c = c - 704
        offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_b_ptr, mean384_b_ptr, rsqrt384_b_ptr, weight384_b_ptr, bias384_b_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    elif pid_c < 92:
        local_c = c - 1088
        offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_c_ptr, mean384_c_ptr, rsqrt384_c_ptr, weight384_c_ptr, bias384_c_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    elif pid_c < 116:
        local_c = c - 1472
        offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x384_d_ptr, mean384_d_ptr, rsqrt384_d_ptr, weight384_d_ptr, bias384_d_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=384, HW_=HW_, W_=W_,
        )
    else:
        local_c = c - 1856
        offsets, x, mean, rsqrt, affine_weight, affine_bias = _load_branch_channels_last(
            x192_ptr, mean192_ptr, rsqrt192_ptr, weight192_ptr, bias192_ptr,
            local_c, n, h, w, mask, c_mask, CBRANCH=192, HW_=HW_, W_=W_,
        )

    sum_masked = tl.load(sum_masked_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_masked_centered = tl.load(sum_masked_centered_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    centered = x - mean[None, :]
    affine = centered * rsqrt[None, :] * affine_weight[None, :] + affine_bias[None, :]
    masked = tl.where(mask & (affine > 0.0), mm * INV_HW_, 0.0)
    variance_term = sum_masked_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_masked * SCALE_
    grad = (masked - centered * variance_term[None, :] - mean_term[None, :]) * (
        rsqrt[None, :] * affine_weight[None, :]
    )
    weight_grad = sum_masked_centered * rsqrt
    first_k_block = tl.program_id(1) == 0

    if pid_c < 20:
        tl.store(grad320_ptr + offsets, grad, mask=mask)
        tl.store(weight_grad320_ptr + local_c, weight_grad, mask=c_mask & first_k_block)
    elif pid_c < 44:
        tl.store(grad384_a_ptr + offsets, grad, mask=mask)
        tl.store(weight_grad384_a_ptr + local_c, weight_grad, mask=c_mask & first_k_block)
    elif pid_c < 68:
        tl.store(grad384_b_ptr + offsets, grad, mask=mask)
        tl.store(weight_grad384_b_ptr + local_c, weight_grad, mask=c_mask & first_k_block)
    elif pid_c < 92:
        tl.store(grad384_c_ptr + offsets, grad, mask=mask)
        tl.store(weight_grad384_c_ptr + local_c, weight_grad, mask=c_mask & first_k_block)
    elif pid_c < 116:
        tl.store(grad384_d_ptr + offsets, grad, mask=mask)
        tl.store(weight_grad384_d_ptr + local_c, weight_grad, mask=c_mask & first_k_block)
    else:
        tl.store(grad192_ptr + offsets, grad, mask=mask)
        tl.store(weight_grad192_ptr + local_c, weight_grad, mask=c_mask & first_k_block)


@oracle_impl(hardware="H100", shapes="(T([128, 2048], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([1, 320, 1, 1], f32), T([1, 320, 1, 1], f32), T([320], f32), T([320], f32), S([128, 2048, 1, 1]), S([128, 2048, 8, 8]))")
def oracle_forward(inputs):
    (
        mm,
        convolution_93,
        getitem_195,
        rsqrt_93,
        primals_564,
        primals_565,
        convolution_92,
        getitem_193,
        rsqrt_92,
        primals_558,
        primals_559,
        convolution_91,
        getitem_191,
        rsqrt_91,
        primals_552,
        primals_553,
        convolution_88,
        getitem_185,
        rsqrt_88,
        primals_534,
        primals_535,
        convolution_87,
        getitem_183,
        rsqrt_87,
        primals_528,
        primals_529,
        convolution_85,
        getitem_179,
        rsqrt_85,
        primals_516,
        primals_517,
        _shape_param_0,
        _shape_param_1,
    ) = inputs
    del _shape_param_0, _shape_param_1

    grad85 = _empty_same_layout(convolution_85)
    weight_grad85 = torch.empty((320,), device=mm.device, dtype=torch.float32)
    grad87 = _empty_same_layout(convolution_87)
    weight_grad87 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad88 = _empty_same_layout(convolution_88)
    weight_grad88 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad91 = _empty_same_layout(convolution_91)
    weight_grad91 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad92 = _empty_same_layout(convolution_92)
    weight_grad92 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad93 = _empty_same_layout(convolution_93)
    weight_grad93 = torch.empty((192,), device=mm.device, dtype=torch.float32)

    block_c = 16
    block_k = 256
    num_k_blocks = triton.cdiv(TOTAL_SPATIAL, block_k)
    sum_masked = torch.empty((MM_C,), device=mm.device, dtype=torch.float32)
    sum_masked_centered = torch.empty((MM_C,), device=mm.device, dtype=torch.float32)
    sum_masked.zero_()
    sum_masked_centered.zero_()

    _all_branches_reduce_split_k_dispatched_kernel[(triton.cdiv(MM_C, block_c), num_k_blocks)](
        mm,
        convolution_85,
        getitem_179,
        rsqrt_85,
        primals_516,
        primals_517,
        convolution_87,
        getitem_183,
        rsqrt_87,
        primals_528,
        primals_529,
        convolution_88,
        getitem_185,
        rsqrt_88,
        primals_534,
        primals_535,
        convolution_91,
        getitem_191,
        rsqrt_91,
        primals_552,
        primals_553,
        convolution_92,
        getitem_193,
        rsqrt_92,
        primals_558,
        primals_559,
        convolution_93,
        getitem_195,
        rsqrt_93,
        primals_564,
        primals_565,
        sum_masked,
        sum_masked_centered,
        MM_C_=MM_C,
        HW_=HW,
        W_=W,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_K=block_k,
        BLOCK_C=block_c,
        num_warps=8,
    )

    _all_branches_epilogue_dispatched_kernel[(triton.cdiv(MM_C, block_c), num_k_blocks)](
        mm,
        convolution_85,
        getitem_179,
        rsqrt_85,
        primals_516,
        primals_517,
        convolution_87,
        getitem_183,
        rsqrt_87,
        primals_528,
        primals_529,
        convolution_88,
        getitem_185,
        rsqrt_88,
        primals_534,
        primals_535,
        convolution_91,
        getitem_191,
        rsqrt_91,
        primals_552,
        primals_553,
        convolution_92,
        getitem_193,
        rsqrt_92,
        primals_558,
        primals_559,
        convolution_93,
        getitem_195,
        rsqrt_93,
        primals_564,
        primals_565,
        sum_masked,
        sum_masked_centered,
        grad85,
        weight_grad85,
        grad87,
        weight_grad87,
        grad88,
        weight_grad88,
        grad91,
        weight_grad91,
        grad92,
        weight_grad92,
        grad93,
        weight_grad93,
        MM_C_=MM_C,
        HW_=HW,
        W_=W,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_K=block_k,
        BLOCK_C=block_c,
        num_warps=8,
    )

    return (
        grad93,
        weight_grad93,
        grad92,
        weight_grad92,
        grad91,
        weight_grad91,
        grad88,
        weight_grad88,
        grad87,
        weight_grad87,
        grad85,
        weight_grad85,
    )


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
