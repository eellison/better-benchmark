"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ViT/DeiT bf16 patch-token plus f32 class-token positional LayerNorm scope in one Triton row kernel, including the channels-last patch view/permute gather, class-token expand, returned f32 virtual-cat tensor, returned f32 positional-add tensor, fp32 population var_mean over hidden dim, eps=1e-6 rsqrt side output, affine scale/bias epilogue, final bf16 cast, and flattened view return for both shape points, whereas Inductor lowers the reshape/permute/expand/cat/add producer, observable statistics, affine cast, and view through generic layout and normalization schedules; Inductor cannot do this today because its LayerNorm scheduler does not sink fixed class-token and patch-token gather producers into a full-scope row-statistics plan while preserving the returned pre-norm tensors and statistic side outputs; the fix is SCHEDULER_FUSION: extend the norm-template scheduler to fold static virtual-cat patch layouts into the row reduction and emit all side outputs plus the bf16 view directly."""

import torch
import triton
import triton.language as tl
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
def _vit_patch_layernorm_kernel(
    patch_ptr,
    class_token_ptr,
    position_ptr,
    weight_ptr,
    bias_ptr,
    cat_out_ptr,
    add_out_ptr,
    mean_out_ptr,
    invstd_out_ptr,
    bf16_out_ptr,
    rows_total: tl.constexpr,
    tokens: tl.constexpr,
    patches: tl.constexpr,
    hidden: tl.constexpr,
    patch_stride_b: tl.constexpr,
    patch_stride_c: tl.constexpr,
    patch_stride_p: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = row_offsets[:, None] < rows_total
    col_mask = cols < hidden
    mask = row_mask & col_mask

    batch = row_offsets[:, None] // tokens
    token = row_offsets[:, None] - batch * tokens
    patch = token - 1
    is_class = token == 0
    offsets = rows * hidden + cols

    class_values = tl.load(
        class_token_ptr + cols,
        mask=col_mask & is_class,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    patch_values = tl.load(
        patch_ptr
        + batch * patch_stride_b
        + patch * patch_stride_p
        + cols * patch_stride_c,
        mask=mask & (token != 0) & (patch < patches),
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    cat_values = tl.where(is_class, class_values, patch_values)
    tl.store(cat_out_ptr + offsets, cat_values, mask=mask)

    position = tl.load(
        position_ptr + token * hidden + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    add_values = _f32_add(cat_values, position)
    tl.store(add_out_ptr + offsets, add_values, mask=mask)

    reduce_input = tl.where(mask, add_values, 0.0)
    mean = _f32_mul(tl.sum(reduce_input, axis=1), 1.0 / hidden)
    centered = _f32_sub(add_values, mean[:, None])
    centered_masked = tl.where(mask, centered, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_masked, centered_masked), axis=1),
        1.0 / hidden,
    )
    invstd = libdevice.rsqrt(_f32_add(variance, eps))

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
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd[:, None]), weight), bias)

    tl.store(mean_out_ptr + row_offsets, mean, mask=row_offsets < rows_total)
    tl.store(invstd_out_ptr + row_offsets, invstd, mask=row_offsets < rows_total)
    tl.store(
        bf16_out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="efb792b2",
    BLOCK_H=256,
    ROW_BLOCK=16,
    num_warps=4,
    num_stages=3,
)
@oracle_impl(
    hardware="B200",
    point="4d589a1c",
    BLOCK_H=1024,
    ROW_BLOCK=1,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    patch_tensor, class_token, position, weight, bias, _patch_view_shape, _expand_shape, view_shape = inputs
    del _patch_view_shape, _expand_shape

    batch = int(patch_tensor.shape[0])
    hidden = int(patch_tensor.shape[1])
    patch_count = int(patch_tensor.shape[2]) * int(patch_tensor.shape[3])
    tokens = patch_count + 1
    rows = batch * tokens
    token_shape = (batch, tokens, hidden)

    cat_out = torch.empty_strided(
        token_shape,
        (tokens * hidden, hidden, 1),
        device=patch_tensor.device,
        dtype=torch.float32,
    )
    add_out = torch.empty_strided(
        token_shape,
        (tokens * hidden, hidden, 1),
        device=patch_tensor.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (batch, tokens, 1),
        (tokens, 1, 1),
        device=patch_tensor.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (batch, tokens, 1),
        (tokens, 1, 1),
        device=patch_tensor.device,
        dtype=torch.float32,
    )
    bf16_base = torch.empty_strided(
        token_shape,
        (tokens * hidden, hidden, 1),
        device=patch_tensor.device,
        dtype=torch.bfloat16,
    )

    _vit_patch_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        patch_tensor,
        class_token,
        position,
        weight,
        bias,
        cat_out,
        add_out,
        mean,
        invstd,
        bf16_base,
        rows_total=rows,
        tokens=tokens,
        patches=patch_count,
        hidden=hidden,
        patch_stride_b=int(patch_tensor.stride(0)),
        patch_stride_c=int(patch_tensor.stride(1)),
        patch_stride_p=int(patch_tensor.stride(3)),
        eps=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return cat_out, add_out, mean, invstd, bf16_base.view(_shape_tuple(view_shape))
