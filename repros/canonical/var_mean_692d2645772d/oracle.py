"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeiT distilled-token patch LayerNorm training scope in one Triton row kernel, including the channels-last bf16 patch view/permute, both expanded f32 token rows, the returned f32 cat and positional-add buffers, fp32 population var_mean over hidden size 768, eps=1e-6 libdevice rsqrt, returned mean and rsqrt side outputs, fp32 affine epilogue, final bf16 cast, and flattened `[25344, 768]` view output, whereas Inductor lowers the decomposed mixed-dtype view/permute/expand/cat/add/var_mean graph through generic producer and normalization schedules; Inductor cannot do this today because fixed-hidden LayerNorm scheduling does not fuse the static DeiT token/patch/position assembly with row statistics while also emitting the required side buffers; the fix is SCHEDULER_FUSION: extend the LayerNorm scheduler to fold these static DeiT token producers and side-output stores into one row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6
BLOCK_H = 1024


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


@triton.autotune(
    configs=[
        triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
    ],
    key=["ROWS", "HIDDEN"],
)
@triton.jit
def _deit_distilled_patch_layernorm_side_kernel(
    patch_ptr,
    class_token_ptr,
    dist_token_ptr,
    pos_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,
    add_ptr,
    mean_ptr,
    rsqrt_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    PATCHES: tl.constexpr,
    HIDDEN: tl.constexpr,
    PATCH_STRIDE_B: tl.constexpr,
    PATCH_STRIDE_P: tl.constexpr,
    PATCH_STRIDE_H: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    row_mask = row_offsets < ROWS
    mask = row_mask[:, None] & (cols < HIDDEN)
    col_mask = cols < HIDDEN

    batch = row_offsets[:, None] // TOKENS
    token = row_offsets[:, None] - batch * TOKENS
    patch = token - 2
    offsets = rows * HIDDEN + cols

    class_values = tl.load(
        class_token_ptr + cols,
        mask=col_mask & (token == 0),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    dist_values = tl.load(
        dist_token_ptr + cols,
        mask=col_mask & (token == 1),
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    patch_values = tl.load(
        patch_ptr + batch * PATCH_STRIDE_B + patch * PATCH_STRIDE_P + cols * PATCH_STRIDE_H,
        mask=mask & (token >= 2) & (patch < PATCHES),
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    pos_values = tl.load(
        pos_ptr + token * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    cat_values = tl.where(
        token == 0,
        class_values,
        tl.where(token == 1, dist_values, patch_values),
    )
    add_values = _f32_add(cat_values, pos_values)
    tl.store(cat_ptr + offsets, cat_values, mask=mask)
    tl.store(add_ptr + offsets, add_values, mask=mask)

    reduce_values = tl.where(mask, add_values, 0.0)
    mean = _f32_mul(tl.sum(reduce_values, axis=1), 1.0 / HIDDEN)[:, None]
    centered = _f32_sub(add_values, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance = _f32_mul(
        tl.sum(_f32_mul(centered_masked, centered_masked), axis=1),
        1.0 / HIDDEN,
    )[:, None]
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    tl.store(mean_ptr + row_offsets[:, None], mean, mask=row_mask[:, None])
    tl.store(rsqrt_ptr + row_offsets[:, None], invstd, mask=row_mask[:, None])

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
    affine = _f32_add(scaled, bias)
    tl.store(
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="1d6386ee", BLOCK=BLOCK_H)
def oracle_forward(inputs, *, BLOCK: int):
    (
        patches,
        class_token,
        dist_token,
        pos_embed,
        weight,
        bias,
        _patch_view_shape,
        _class_expand_shape,
        _dist_expand_shape,
        output_view_shape,
    ) = inputs

    batch = int(patches.shape[0])
    hidden = int(weight.shape[0])
    patch_count = int(patches.shape[2]) * int(patches.shape[3])
    tokens = patch_count + 2
    rows = batch * tokens
    base_shape = (batch, tokens, hidden)
    base_stride = _contiguous_stride(base_shape)
    side_shape = (batch, tokens, 1)

    cat = torch.empty_strided(
        base_shape,
        base_stride,
        device=patches.device,
        dtype=torch.float32,
    )
    add = torch.empty_strided(
        base_shape,
        base_stride,
        device=patches.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        side_shape,
        (tokens, 1, 1),
        device=patches.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        side_shape,
        (tokens, 1, 1),
        device=patches.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        base_shape,
        base_stride,
        device=patches.device,
        dtype=torch.bfloat16,
    )

    grid = lambda meta: (triton.cdiv(rows, meta["ROW_BLOCK"]),)
    _deit_distilled_patch_layernorm_side_kernel[grid](
        patches,
        class_token,
        dist_token,
        pos_embed,
        weight,
        bias,
        cat,
        add,
        mean,
        rsqrt,
        out,
        ROWS=rows,
        TOKENS=tokens,
        PATCHES=patch_count,
        HIDDEN=hidden,
        PATCH_STRIDE_B=patches.stride(0),
        PATCH_STRIDE_P=patches.stride(3),
        PATCH_STRIDE_H=patches.stride(1),
        EPSILON=EPS,
        BLOCK=BLOCK,
    )

    return cat, add, mean, rsqrt, out.view(_as_shape(output_view_shape))
