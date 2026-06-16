"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ViT class-token/patch-token bf16 LayerNorm scope by directly gathering the expanded class token or flattened patch token, applying the bf16 positional add once, storing the returned bf16 add tensor, and keeping the bf16-rounded row live for the fp32 population var_mean, eps=1e-6 rsqrt, bf16 affine epilogue, and final flattened view output, whereas Inductor lowers the expand/view/permute/cat/add side output and the downstream LayerNorm through generic layout and normalization schedules; Inductor cannot do this today because its scheduler/codegen does not form one full-scope fixed-hidden LayerNorm lowering that both materializes a required bf16 producer side output and reuses it for the dependent row reduction/affine store; the fix is SCHEDULER_FUSION: add a ViT patch LayerNorm schedule that fuses direct class/patch gather, explicit bf16 add materialization, and the dependent normalization epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6


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
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _vit_patch_layernorm_kernel(
    class_token_ptr,
    conv_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    CONV_STRIDE_B: tl.constexpr,
    CONV_STRIDE_C: tl.constexpr,
    CONV_STRIDE_P: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]

    token = rows % TOKENS
    batch = rows // TOKENS
    is_class = token == 0
    patch = token - 1

    class_value = tl.load(
        class_token_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)[None, :]
    patch_value = tl.load(
        conv_ptr
        + batch[:, None] * CONV_STRIDE_B
        + cols[None, :] * CONV_STRIDE_C
        + patch[:, None] * CONV_STRIDE_P,
        mask=mask & (token[:, None] != 0),
        other=0.0,
    ).to(tl.float32)
    pos = tl.load(
        pos_ptr + token[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x_unrounded = _f32_add(tl.where(is_class[:, None], class_value, patch_value), pos)
    x = _round_bf16_to_f32(x_unrounded)
    tl.store(add_out_ptr + rows[:, None] * HIDDEN + cols[None, :], x, mask=mask)
    mean_acc = tl.zeros([BLOCK_R, BLOCK_C], tl.float32)
    m2_acc = tl.zeros([BLOCK_R, BLOCK_C], tl.float32)
    weight_acc = tl.zeros([BLOCK_R, BLOCK_C], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    centered = _f32_sub(x, mean[:, None])
    variance = m2 / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))

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
    y = _f32_add(
        _f32_mul(_f32_mul(centered, invstd[:, None]), weight[None, :]),
        bias[None, :],
    ).to(tl.bfloat16)
    tl.store(norm_out_ptr + rows[:, None] * HIDDEN + cols[None, :], y, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(
    hardware="B200",
    point="b1389f17",
    BLOCK_R=4,
    BLOCK_C=256,
    num_warps=4,
    num_stages=3,
)
@oracle_impl(
    hardware="B200",
    point="5b1eacf5",
    BLOCK_R=1,
    BLOCK_C=1024,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    num_warps: int,
    num_stages: int,
):
    class_token, conv, pos, weight, bias, _expand_shape, _view_shape, out_shape = inputs
    batch = int(conv.shape[0])
    hidden = int(weight.shape[0])
    tokens = int(pos.shape[1])
    rows = batch * tokens
    add_shape = (batch, tokens, hidden)
    norm_shape = tuple(int(dim) for dim in out_shape)
    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    _vit_patch_layernorm_kernel[(triton.cdiv(rows, BLOCK_R),)](
        class_token,
        conv,
        pos,
        weight,
        bias,
        add_out,
        norm_out,
        ROWS=rows,
        TOKENS=tokens,
        HIDDEN=hidden,
        CONV_STRIDE_B=int(conv.stride(0)),
        CONV_STRIDE_C=int(conv.stride(1)),
        CONV_STRIDE_P=int(conv.stride(3)),
        EPS_=EPS,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
