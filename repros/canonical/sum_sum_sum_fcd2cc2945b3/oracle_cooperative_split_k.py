"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle assigns each channel slice of all six Inception batch-norm-backward branches to a full-scope fused producer that reduces the shared `(N, H, W)` domain and immediately writes the complete twelve-output `Repro.forward` tuple with each dependent input-gradient epilogue and scale-gradient vector, whereas Inductor currently decomposes the `mm / 64` channel slices, ReLU masks, sibling `sum([0, 2, 3])` reductions, and BN-backward epilogues into separate pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates many disjoint channel-slice BN branches with finalized channel summaries feeding full-tensor side-output epilogues; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible channel reductions across the `(N, H, W)` K dimension, accumulate multiple summaries per branch, and fuse the dependent gradient/vector epilogues for the complete return tuple."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_fcd2cc2945b3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
MM_C = 2048
H = 8
W = 8
HW = H * W
TOTAL_SPATIAL = N * HW
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL



def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _masked_bn_epilogue_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    grad_ptr,
    weight_grad_ptr,
    C_: tl.constexpr,
    MM_C_: tl.constexpr,
    MM_OFFSET_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + n * MM_C_ + MM_OFFSET_ + c, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_masked = tl.load(sum_masked_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_masked_centered = tl.load(sum_masked_centered_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * affine_weight + affine_bias
    masked = tl.where(affine > 0.0, mm * INV_HW_, 0.0)

    variance_term = sum_masked_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_masked * SCALE_
    grad = (masked - centered * variance_term - mean_term) * (rsqrt * affine_weight)

    tl.store(grad_ptr + offsets, grad, mask=mask)
    tl.store(weight_grad_ptr + c, sum_masked_centered * rsqrt, mask=mask & (n == 0) & (hw == 0))


def _oracle_branch_epilogue(
    mm: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rsqrt: torch.Tensor,
    affine_weight: torch.Tensor,
    affine_bias: torch.Tensor,
    sum_masked: torch.Tensor,
    sum_masked_centered: torch.Tensor,
    mm_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = x.shape[1]
    grad = torch.empty_like(x)
    weight_grad = torch.empty((c,), device=x.device, dtype=torch.float32)
    block_elems = 256
    numel = N * c * HW
    _masked_bn_epilogue_kernel[(triton.cdiv(numel, block_elems),)](
        mm,
        x,
        mean,
        rsqrt,
        affine_weight,
        affine_bias,
        sum_masked,
        sum_masked_centered,
        grad,
        weight_grad,
        C_=c,
        MM_C_=MM_C,
        MM_OFFSET_=mm_offset,
        HW_=HW,
        NUMEL_=numel,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )
    return grad, weight_grad


@triton.jit
def _all_branches_reduce_split_k_kernel(
    mm_ptr,
    x515_ptr,
    mean516_ptr,
    rsqrt517_ptr,
    weight212_ptr,
    bias213_ptr,
    x521_ptr,
    mean522_ptr,
    rsqrt523_ptr,
    weight217_ptr,
    bias218_ptr,
    x524_ptr,
    mean525_ptr,
    rsqrt526_ptr,
    weight220_ptr,
    bias221_ptr,
    x533_ptr,
    mean534_ptr,
    rsqrt535_ptr,
    weight227_ptr,
    bias228_ptr,
    x536_ptr,
    mean537_ptr,
    rsqrt538_ptr,
    weight230_ptr,
    bias231_ptr,
    x540_ptr,
    mean541_ptr,
    rsqrt542_ptr,
    weight233_ptr,
    bias234_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)

    c_mask = c < MM_C_
    k_mask = k < TOTAL_SPATIAL_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    mm = tl.load(mm_ptr + n[None, :] * MM_C_ + c[:, None], mask=mask, other=0.0).to(tl.float32)

    b515 = c < 320
    b521 = (c >= 320) & (c < 704)
    b524 = (c >= 704) & (c < 1088)
    b533 = (c >= 1088) & (c < 1472)
    b536 = (c >= 1472) & (c < 1856)
    b540 = c >= 1856

    x = tl.full((BLOCK_C, BLOCK_K), 0.0, tl.float32)
    mean = tl.full((BLOCK_C,), 0.0, tl.float32)
    rsqrt = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_weight = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_bias = tl.full((BLOCK_C,), 0.0, tl.float32)

    c515 = tl.where(b515, c, 0)
    mask515 = mask & b515[:, None]
    x += tl.load(x515_ptr + n[None, :] * (320 * HW_) + c515[:, None] * HW_ + hw[None, :], mask=mask515, other=0.0).to(tl.float32)
    mean += tl.load(mean516_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt517_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight212_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias213_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)

    c521 = tl.where(b521, c - 320, 0)
    mask521 = mask & b521[:, None]
    x += tl.load(x521_ptr + n[None, :] * (384 * HW_) + c521[:, None] * HW_ + hw[None, :], mask=mask521, other=0.0).to(tl.float32)
    mean += tl.load(mean522_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt523_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight217_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias218_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)

    c524 = tl.where(b524, c - 704, 0)
    mask524 = mask & b524[:, None]
    x += tl.load(x524_ptr + n[None, :] * (384 * HW_) + c524[:, None] * HW_ + hw[None, :], mask=mask524, other=0.0).to(tl.float32)
    mean += tl.load(mean525_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt526_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight220_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias221_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)

    c533 = tl.where(b533, c - 1088, 0)
    mask533 = mask & b533[:, None]
    x += tl.load(x533_ptr + n[None, :] * (384 * HW_) + c533[:, None] * HW_ + hw[None, :], mask=mask533, other=0.0).to(tl.float32)
    mean += tl.load(mean534_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt535_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight227_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias228_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)

    c536 = tl.where(b536, c - 1472, 0)
    mask536 = mask & b536[:, None]
    x += tl.load(x536_ptr + n[None, :] * (384 * HW_) + c536[:, None] * HW_ + hw[None, :], mask=mask536, other=0.0).to(tl.float32)
    mean += tl.load(mean537_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt538_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight230_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias231_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)

    c540 = tl.where(b540, c - 1856, 0)
    mask540 = mask & b540[:, None]
    x += tl.load(x540_ptr + n[None, :] * (192 * HW_) + c540[:, None] * HW_ + hw[None, :], mask=mask540, other=0.0).to(tl.float32)
    mean += tl.load(mean541_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt542_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight233_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias234_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)

    centered = x - mean[:, None]
    affine = centered * rsqrt[:, None] * affine_weight[:, None] + affine_bias[:, None]
    masked = tl.where(affine > 0.0, mm * INV_HW_, 0.0)

    tl.atomic_add(
        sum_masked_ptr + c,
        tl.sum(masked, axis=1),
        sem="relaxed",
        mask=c_mask,
    )
    tl.atomic_add(
        sum_masked_centered_ptr + c,
        tl.sum(masked * centered, axis=1),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _all_branches_epilogue_kernel(
    mm_ptr,
    x515_ptr,
    mean516_ptr,
    rsqrt517_ptr,
    weight212_ptr,
    bias213_ptr,
    x521_ptr,
    mean522_ptr,
    rsqrt523_ptr,
    weight217_ptr,
    bias218_ptr,
    x524_ptr,
    mean525_ptr,
    rsqrt526_ptr,
    weight220_ptr,
    bias221_ptr,
    x533_ptr,
    mean534_ptr,
    rsqrt535_ptr,
    weight227_ptr,
    bias228_ptr,
    x536_ptr,
    mean537_ptr,
    rsqrt538_ptr,
    weight230_ptr,
    bias231_ptr,
    x540_ptr,
    mean541_ptr,
    rsqrt542_ptr,
    weight233_ptr,
    bias234_ptr,
    sum_masked_ptr,
    sum_masked_centered_ptr,
    grad515_ptr,
    weight_grad515_ptr,
    grad521_ptr,
    weight_grad521_ptr,
    grad524_ptr,
    weight_grad524_ptr,
    grad533_ptr,
    weight_grad533_ptr,
    grad536_ptr,
    weight_grad536_ptr,
    grad540_ptr,
    weight_grad540_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)

    c_mask = c < MM_C_
    k_mask = k < TOTAL_SPATIAL_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    mm = tl.load(mm_ptr + n[None, :] * MM_C_ + c[:, None], mask=mask, other=0.0).to(tl.float32)

    b515 = c < 320
    b521 = (c >= 320) & (c < 704)
    b524 = (c >= 704) & (c < 1088)
    b533 = (c >= 1088) & (c < 1472)
    b536 = (c >= 1472) & (c < 1856)
    b540 = c >= 1856

    x = tl.full((BLOCK_C, BLOCK_K), 0.0, tl.float32)
    mean = tl.full((BLOCK_C,), 0.0, tl.float32)
    rsqrt = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_weight = tl.full((BLOCK_C,), 0.0, tl.float32)
    affine_bias = tl.full((BLOCK_C,), 0.0, tl.float32)

    c515 = tl.where(b515, c, 0)
    mask515 = mask & b515[:, None]
    offsets515 = n[None, :] * (320 * HW_) + c515[:, None] * HW_ + hw[None, :]
    x += tl.load(x515_ptr + offsets515, mask=mask515, other=0.0).to(tl.float32)
    mean += tl.load(mean516_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt517_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight212_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias213_ptr + c515, mask=c_mask & b515, other=0.0).to(tl.float32)

    c521 = tl.where(b521, c - 320, 0)
    mask521 = mask & b521[:, None]
    offsets521 = n[None, :] * (384 * HW_) + c521[:, None] * HW_ + hw[None, :]
    x += tl.load(x521_ptr + offsets521, mask=mask521, other=0.0).to(tl.float32)
    mean += tl.load(mean522_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt523_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight217_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias218_ptr + c521, mask=c_mask & b521, other=0.0).to(tl.float32)

    c524 = tl.where(b524, c - 704, 0)
    mask524 = mask & b524[:, None]
    offsets524 = n[None, :] * (384 * HW_) + c524[:, None] * HW_ + hw[None, :]
    x += tl.load(x524_ptr + offsets524, mask=mask524, other=0.0).to(tl.float32)
    mean += tl.load(mean525_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt526_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight220_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias221_ptr + c524, mask=c_mask & b524, other=0.0).to(tl.float32)

    c533 = tl.where(b533, c - 1088, 0)
    mask533 = mask & b533[:, None]
    offsets533 = n[None, :] * (384 * HW_) + c533[:, None] * HW_ + hw[None, :]
    x += tl.load(x533_ptr + offsets533, mask=mask533, other=0.0).to(tl.float32)
    mean += tl.load(mean534_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt535_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight227_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias228_ptr + c533, mask=c_mask & b533, other=0.0).to(tl.float32)

    c536 = tl.where(b536, c - 1472, 0)
    mask536 = mask & b536[:, None]
    offsets536 = n[None, :] * (384 * HW_) + c536[:, None] * HW_ + hw[None, :]
    x += tl.load(x536_ptr + offsets536, mask=mask536, other=0.0).to(tl.float32)
    mean += tl.load(mean537_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt538_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight230_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias231_ptr + c536, mask=c_mask & b536, other=0.0).to(tl.float32)

    c540 = tl.where(b540, c - 1856, 0)
    mask540 = mask & b540[:, None]
    offsets540 = n[None, :] * (192 * HW_) + c540[:, None] * HW_ + hw[None, :]
    x += tl.load(x540_ptr + offsets540, mask=mask540, other=0.0).to(tl.float32)
    mean += tl.load(mean541_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt542_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight233_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias234_ptr + c540, mask=c_mask & b540, other=0.0).to(tl.float32)

    sum_masked = tl.load(sum_masked_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_masked_centered = tl.load(sum_masked_centered_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[:, None]
    affine = centered * rsqrt[:, None] * affine_weight[:, None] + affine_bias[:, None]
    masked = tl.where(affine > 0.0, mm * INV_HW_, 0.0)

    variance_term = sum_masked_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_masked * SCALE_
    grad = (masked - centered * variance_term[:, None] - mean_term[:, None]) * (
        rsqrt[:, None] * affine_weight[:, None]
    )
    first_k_block = tl.program_id(1) == 0
    weight_grad = sum_masked_centered * rsqrt

    tl.store(grad515_ptr + offsets515, grad, mask=mask515)
    tl.store(weight_grad515_ptr + c515, weight_grad, mask=(c_mask & b515) & first_k_block)

    tl.store(grad521_ptr + offsets521, grad, mask=mask521)
    tl.store(weight_grad521_ptr + c521, weight_grad, mask=(c_mask & b521) & first_k_block)

    tl.store(grad524_ptr + offsets524, grad, mask=mask524)
    tl.store(weight_grad524_ptr + c524, weight_grad, mask=(c_mask & b524) & first_k_block)

    tl.store(grad533_ptr + offsets533, grad, mask=mask533)
    tl.store(weight_grad533_ptr + c533, weight_grad, mask=(c_mask & b533) & first_k_block)

    tl.store(grad536_ptr + offsets536, grad, mask=mask536)
    tl.store(weight_grad536_ptr + c536, weight_grad, mask=(c_mask & b536) & first_k_block)

    tl.store(grad540_ptr + offsets540, grad, mask=mask540)
    tl.store(weight_grad540_ptr + c540, weight_grad, mask=(c_mask & b540) & first_k_block)


@triton.jit
def _all_branches_single_pass_kernel(
    mm_ptr,
    x515_ptr,
    mean516_ptr,
    rsqrt517_ptr,
    weight212_ptr,
    bias213_ptr,
    x521_ptr,
    mean522_ptr,
    rsqrt523_ptr,
    weight217_ptr,
    bias218_ptr,
    x524_ptr,
    mean525_ptr,
    rsqrt526_ptr,
    weight220_ptr,
    bias221_ptr,
    x533_ptr,
    mean534_ptr,
    rsqrt535_ptr,
    weight227_ptr,
    bias228_ptr,
    x536_ptr,
    mean537_ptr,
    rsqrt538_ptr,
    weight230_ptr,
    bias231_ptr,
    x540_ptr,
    mean541_ptr,
    rsqrt542_ptr,
    weight233_ptr,
    bias234_ptr,
    grad515_ptr,
    weight_grad515_ptr,
    grad521_ptr,
    weight_grad521_ptr,
    grad524_ptr,
    weight_grad524_ptr,
    grad533_ptr,
    weight_grad533_ptr,
    grad536_ptr,
    weight_grad536_ptr,
    grad540_ptr,
    weight_grad540_ptr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
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
    mm = tl.load(mm_ptr + n * MM_C_ + c, mask=k_mask, other=0.0).to(tl.float32)

    b515 = c < 320
    b521 = (c >= 320) & (c < 704)
    b524 = (c >= 704) & (c < 1088)
    b533 = (c >= 1088) & (c < 1472)
    b536 = (c >= 1472) & (c < 1856)
    b540 = c >= 1856

    local_c = tl.where(
        b515,
        c,
        tl.where(
            b521,
            c - 320,
            tl.where(
                b524,
                c - 704,
                tl.where(b533, c - 1088, tl.where(b536, c - 1472, c - 1856)),
            ),
        ),
    )

    x = tl.full((BLOCK_K,), 0.0, tl.float32)
    mean = tl.full((), 0.0, tl.float32)
    rsqrt = tl.full((), 0.0, tl.float32)
    affine_weight = tl.full((), 0.0, tl.float32)
    affine_bias = tl.full((), 0.0, tl.float32)

    offsets515 = n * (320 * HW_) + local_c * HW_ + hw
    x += tl.load(x515_ptr + offsets515, mask=k_mask & b515, other=0.0).to(tl.float32)
    mean += tl.load(mean516_ptr + local_c, mask=b515, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt517_ptr + local_c, mask=b515, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight212_ptr + local_c, mask=b515, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias213_ptr + local_c, mask=b515, other=0.0).to(tl.float32)

    offsets521 = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x521_ptr + offsets521, mask=k_mask & b521, other=0.0).to(tl.float32)
    mean += tl.load(mean522_ptr + local_c, mask=b521, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt523_ptr + local_c, mask=b521, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight217_ptr + local_c, mask=b521, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias218_ptr + local_c, mask=b521, other=0.0).to(tl.float32)

    offsets524 = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x524_ptr + offsets524, mask=k_mask & b524, other=0.0).to(tl.float32)
    mean += tl.load(mean525_ptr + local_c, mask=b524, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt526_ptr + local_c, mask=b524, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight220_ptr + local_c, mask=b524, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias221_ptr + local_c, mask=b524, other=0.0).to(tl.float32)

    offsets533 = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x533_ptr + offsets533, mask=k_mask & b533, other=0.0).to(tl.float32)
    mean += tl.load(mean534_ptr + local_c, mask=b533, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt535_ptr + local_c, mask=b533, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight227_ptr + local_c, mask=b533, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias228_ptr + local_c, mask=b533, other=0.0).to(tl.float32)

    offsets536 = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x536_ptr + offsets536, mask=k_mask & b536, other=0.0).to(tl.float32)
    mean += tl.load(mean537_ptr + local_c, mask=b536, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt538_ptr + local_c, mask=b536, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight230_ptr + local_c, mask=b536, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias231_ptr + local_c, mask=b536, other=0.0).to(tl.float32)

    offsets540 = n * (192 * HW_) + local_c * HW_ + hw
    x += tl.load(x540_ptr + offsets540, mask=k_mask & b540, other=0.0).to(tl.float32)
    mean += tl.load(mean541_ptr + local_c, mask=b540, other=0.0).to(tl.float32)
    rsqrt += tl.load(rsqrt542_ptr + local_c, mask=b540, other=0.0).to(tl.float32)
    affine_weight += tl.load(weight233_ptr + local_c, mask=b540, other=0.0).to(tl.float32)
    affine_bias += tl.load(bias234_ptr + local_c, mask=b540, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * affine_weight + affine_bias
    masked = tl.where(k_mask & (affine > 0.0), mm * INV_HW_, 0.0)
    sum_masked = tl.sum(masked, axis=0)
    sum_masked_centered = tl.sum(masked * centered, axis=0)

    variance_term = sum_masked_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_masked * SCALE_
    grad = (masked - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    weight_grad = sum_masked_centered * rsqrt

    tl.store(grad515_ptr + offsets515, grad, mask=k_mask & b515)
    tl.store(weight_grad515_ptr + local_c, weight_grad, mask=b515)

    tl.store(grad521_ptr + offsets521, grad, mask=k_mask & b521)
    tl.store(weight_grad521_ptr + local_c, weight_grad, mask=b521)

    tl.store(grad524_ptr + offsets524, grad, mask=k_mask & b524)
    tl.store(weight_grad524_ptr + local_c, weight_grad, mask=b524)

    tl.store(grad533_ptr + offsets533, grad, mask=k_mask & b533)
    tl.store(weight_grad533_ptr + local_c, weight_grad, mask=b533)

    tl.store(grad536_ptr + offsets536, grad, mask=k_mask & b536)
    tl.store(weight_grad536_ptr + local_c, weight_grad, mask=b536)

    tl.store(grad540_ptr + offsets540, grad, mask=k_mask & b540)
    tl.store(weight_grad540_ptr + local_c, weight_grad, mask=b540)

def oracle_full(
    mm: torch.Tensor,
    arg540_1: torch.Tensor,
    arg541_1: torch.Tensor,
    arg542_1: torch.Tensor,
    arg233_1: torch.Tensor,
    arg234_1: torch.Tensor,
    arg536_1: torch.Tensor,
    arg537_1: torch.Tensor,
    arg538_1: torch.Tensor,
    arg230_1: torch.Tensor,
    arg231_1: torch.Tensor,
    arg533_1: torch.Tensor,
    arg534_1: torch.Tensor,
    arg535_1: torch.Tensor,
    arg227_1: torch.Tensor,
    arg228_1: torch.Tensor,
    arg524_1: torch.Tensor,
    arg525_1: torch.Tensor,
    arg526_1: torch.Tensor,
    arg220_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg521_1: torch.Tensor,
    arg522_1: torch.Tensor,
    arg523_1: torch.Tensor,
    arg217_1: torch.Tensor,
    arg218_1: torch.Tensor,
    arg515_1: torch.Tensor,
    arg516_1: torch.Tensor,
    arg517_1: torch.Tensor,
    arg212_1: torch.Tensor,
    arg213_1: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, ...]:
    del _shape_param_0, _shape_param_1

    assert mm.shape == (N, MM_C)
    assert arg540_1.shape == (N, 192, H, W)
    assert arg536_1.shape == (N, 384, H, W)
    assert arg533_1.shape == (N, 384, H, W)
    assert arg524_1.shape == (N, 384, H, W)
    assert arg521_1.shape == (N, 384, H, W)
    assert arg515_1.shape == (N, 320, H, W)
    assert mm.is_contiguous()
    assert arg540_1.is_contiguous()
    assert arg536_1.is_contiguous()
    assert arg533_1.is_contiguous()
    assert arg524_1.is_contiguous()
    assert arg521_1.is_contiguous()
    assert arg515_1.is_contiguous()

    grad515 = torch.empty_like(arg515_1)
    weight_grad515 = torch.empty((320,), device=mm.device, dtype=torch.float32)
    grad521 = torch.empty_like(arg521_1)
    weight_grad521 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad524 = torch.empty_like(arg524_1)
    weight_grad524 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad533 = torch.empty_like(arg533_1)
    weight_grad533 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad536 = torch.empty_like(arg536_1)
    weight_grad536 = torch.empty((384,), device=mm.device, dtype=torch.float32)
    grad540 = torch.empty_like(arg540_1)
    weight_grad540 = torch.empty((192,), device=mm.device, dtype=torch.float32)

    _all_branches_single_pass_kernel[(MM_C,)](
        mm,
        arg515_1,
        arg516_1,
        arg517_1,
        arg212_1,
        arg213_1,
        arg521_1,
        arg522_1,
        arg523_1,
        arg217_1,
        arg218_1,
        arg524_1,
        arg525_1,
        arg526_1,
        arg220_1,
        arg221_1,
        arg533_1,
        arg534_1,
        arg535_1,
        arg227_1,
        arg228_1,
        arg536_1,
        arg537_1,
        arg538_1,
        arg230_1,
        arg231_1,
        arg540_1,
        arg541_1,
        arg542_1,
        arg233_1,
        arg234_1,
        grad515,
        weight_grad515,
        grad521,
        weight_grad521,
        grad524,
        weight_grad524,
        grad533,
        weight_grad533,
        grad536,
        weight_grad536,
        grad540,
        weight_grad540,
        MM_C_=MM_C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_K=TOTAL_SPATIAL,
        num_warps=8,
    )

    return (
        grad540,
        weight_grad540,
        grad536,
        weight_grad536,
        grad533,
        weight_grad533,
        grad524,
        weight_grad524,
        grad521,
        weight_grad521,
        grad515,
        weight_grad515,
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_forward(inputs):
    return oracle_full(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
