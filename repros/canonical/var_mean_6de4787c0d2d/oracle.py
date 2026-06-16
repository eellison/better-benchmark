"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Swin patch-merge residual add, fixed 2x2 layout clone/view, fp32 population `var_mean(..., dim=3, correction=0, keepdim=True)`, eps-before-rsqrt affine epilogue, final bf16 cast, and contiguous flattened output in one Triton row kernel, whereas Inductor lowers the residual add and patch-merge clone/view as separate producer/layout work before scheduling the generic normalization template; Inductor cannot do this today because the norm-template scheduler does not recognize the deterministic Swin patch-merge reshape/permute/clone producer as a direct row source while preserving the bf16 add and final-cast boundaries; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to sink fixed patch-merge layout indexing and residual-add producers into the row-wise normalization load plan and emit the flattened bf16 output directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _swin_patchmerge_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    INPUT_TOKENS: tl.constexpr,
    INPUT_CHANNELS: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    MERGED_WIDTH: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask

    channel_group = cols // INPUT_CHANNELS
    channel = cols - channel_group * INPUT_CHANNELS
    inner_w = channel_group // 2
    inner_h = channel_group - inner_w * 2

    merged_w = row_ids % MERGED_WIDTH
    tmp = row_ids // MERGED_WIDTH
    merged_h = tmp % MERGED_WIDTH
    batch = tmp // MERGED_WIDTH

    source_h = merged_h * 2 + inner_h
    source_w = merged_w * 2 + inner_w
    source_token = source_h * INPUT_WIDTH + source_w
    source_offsets = batch * INPUT_TOKENS * INPUT_CHANNELS + source_token * INPUT_CHANNELS + channel

    flat = tl.load(
        flat_ptr + source_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + source_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(residual, flat).to(tl.bfloat16).to(tl.float32)

    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1)[:, None] / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    y = _f32_add(scaled, bias)

    out_offsets = row_ids * HIDDEN + cols
    tl.store(out_ptr + out_offsets, y.to(tl.bfloat16), mask=mask)


# 6ba63620: (T([25088,512], bf16), T([128,196,512], bf16), T([2048], bf16), T([2048], bf16), ...)
@oracle_impl(hardware="B200", point="6ba63620", BLOCK_H=2048, ROW_BLOCK=1, num_warps=8, num_stages=3)
# d431181f: (T([100352,256], bf16), T([128,784,256], bf16), T([1024], bf16), T([1024], bf16), ...)
@oracle_impl(hardware="B200", point="d431181f", BLOCK_H=1024, ROW_BLOCK=2, num_warps=4, num_stages=3)
# 2eb9cc7a: (T([401408,128], bf16), T([128,3136,128], bf16), T([512], bf16), T([512], bf16), ...)
@oracle_impl(hardware="B200", point="2eb9cc7a", BLOCK_H=512, ROW_BLOCK=4, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, shape2, _shape3, shape4 = inputs
    height = int(shape1[1])
    width = int(shape1[2])
    channels = int(shape1[3])
    merged_width = int(shape2[3])
    rows = int(shape4[0])
    hidden = int(shape4[1])

    out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _swin_patchmerge_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        ROWS=rows,
        INPUT_TOKENS=height * width,
        INPUT_CHANNELS=channels,
        INPUT_WIDTH=width,
        MERGED_WIDTH=merged_width,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
