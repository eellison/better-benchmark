"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DINOv2 bf16-input residual LayerNorm scope in one Triton row kernel, including the `[175360,768] -> [128,1370,768]` view, bf16-to-fp32 gamma multiply, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, returned normalized f32 tensor, returned all-row `rsqrt / 768` side output, and only the token-0 affine f32 clone, whereas Inductor schedules the affine LayerNorm epilogue across all `[128,1370,768]` rows before applying `select(dim=1, index=0).clone()`; Inductor cannot do this today because its normalization scheduler does not sink a constant token select through row-independent affine LayerNorm while another returned output still needs the all-row normalized tensor and inverse scale; the fix is ALGEBRAIC_ELIMINATION: split the LayerNorm epilogue so all rows emit required statistics/normalized outputs but affine scale/bias work is narrowed to the selected token rows."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 175360
BATCH = 128
TOKENS = 1370
HIDDEN = 768
BLOCK_H = 1024
ACT_SHAPE = (BATCH, TOKENS, HIDDEN)
ACT_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
SELECTED_SHAPE = (BATCH, HIDDEN)
SELECTED_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, TOKENS, 1)
SIDE_STRIDE = (TOKENS, 1, 1)


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
        triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
    ],
    key=["ROWS", "HIDDEN"],
)
@triton.jit
def _dinov2_norm_selected_affine_kernel(
    input_bf16_ptr,
    gamma_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_out_ptr,
    selected_out_ptr,
    side_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]
    mask = (row_ids[:, None] < ROWS) & (cols[None, :] < HIDDEN)
    col_mask = cols < HIDDEN

    input_bf16 = tl.load(input_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = _f32_mul(input_bf16, gamma[None, :])
    x = _f32_add(residual, scaled)

    mean_acc = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
    m2_acc = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
    weight_acc = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
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
    variance = m2 / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-6))

    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    tl.store(norm_out_ptr + offsets, normalized, mask=mask)
    tl.store(side_out_ptr + row_ids, _f32_mul(invstd, 0.0013020833333333333), mask=row_ids < ROWS)

    token = row_ids - (row_ids // TOKENS) * TOKENS
    selected = token == 0
    affine_weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, affine_weight[None, :]), affine_bias[None, :])
    batch = row_ids // TOKENS
    selected_offsets = batch[:, None] * HIDDEN + cols[None, :]
    tl.store(selected_out_ptr + selected_offsets, affine, mask=mask & selected[:, None])


# 70d2be15: (T([175360,768], bf16), T([768], f32), T([128,1370,768], f32), ...)
@oracle_impl(hardware="B200", point="70d2be15", BLOCK_H=BLOCK_H)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0 = inputs
    norm_shape = tuple(int(dim) for dim in shape0)
    norm_out = torch.empty_strided(
        norm_shape,
        ACT_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    selected_out = torch.empty_strided(
        SELECTED_SHAPE,
        SELECTED_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    side_out = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _dinov2_norm_selected_affine_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        norm_out,
        selected_out,
        side_out,
        ROWS=ROWS,
        HIDDEN=HIDDEN,
        TOKENS=TOKENS,
        BLOCK_H=BLOCK_H,
    )
    return norm_out, selected_out, side_out
