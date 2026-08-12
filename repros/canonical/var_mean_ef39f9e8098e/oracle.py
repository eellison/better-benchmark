"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DINOv2 bf16-input affine-residual LayerNorm scope in one fixed-hidden Triton row kernel, including the `[175360,768] -> [128,1370,768]` view, bf16-to-fp32 gamma multiply, residual add returned as `add`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 `libdevice.rsqrt`, returned normalized tensor, bf16 affine output view, and sibling `rsqrt / 768` output, whereas Inductor lowers the captured graph through its generic normalization scheduler for the reduction plus visible multi-output stores; Inductor cannot do this today because the scheduler does not specialize this four-output fixed-hidden residual LayerNorm pattern while retaining the producer tile and exact bf16/fp32 rounding boundaries across all returned tensors; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to fuse the bf16 producer, residual add side output, row statistics, normalized side output, affine bf16 store, and inverse-std side output into one guarded DINOv2 LayerNorm schedule."""

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
EPS = 1.0e-6
INV_HIDDEN = 0.0013020833333333333
ACT_SHAPE = (BATCH, TOKENS, HIDDEN)
ACT_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
FLAT_SHAPE = (ROWS, HIDDEN)
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
def _dinov2_layernorm_four_output_kernel(
    input_bf16_ptr,
    gamma_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    bf16_out_ptr,
    side_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
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
    tl.store(add_out_ptr + offsets, x, mask=mask)

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

    affine_weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scaled_norm = _f32_mul(normalized, affine_weight[None, :])
    affine = _f32_add(scaled_norm, affine_bias[None, :])
    tl.store(bf16_out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)

    side = _f32_mul(invstd, 0.0013020833333333333)
    tl.store(side_out_ptr + row_ids, side, mask=row_ids < ROWS)


# 70d2be15: (T([175360,768], bf16), T([768], f32), T([128,1370,768], f32), ...)
@oracle_impl(hardware="B200", point="70d2be15", BLOCK_H=BLOCK_H)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    add_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_0),
        ACT_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    norm_out = torch.empty_strided(
        ACT_SHAPE,
        ACT_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    side_out = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=arg0_1.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _dinov2_layernorm_four_output_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        add_out,
        norm_out,
        bf16_out,
        side_out,
        ROWS=ROWS,
        HIDDEN=HIDDEN,
        BLOCK_H=BLOCK_H,
    )
    return add_out, norm_out, bf16_out, side_out
