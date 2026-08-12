"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet bf16 channel-cat, fp32 BN-inference affine, explicit bf16 output cast, and bf16 ReLU scope by keeping the concat virtual and hoisting the per-channel reciprocal-sqrt terms, whereas Inductor schedules the fixed channel concat and downstream affine/ReLU pointwise work through generic materialization and broadcast kernels; Inductor cannot do this today because its scheduler does not inline fixed-shape channel concat producers into channel-dependent pointwise consumers while preserving the captured fp32 affine order, bf16 cast boundary, NaN behavior, and contiguous output layout; the fix is SCHEDULER_FUSION: teach pointwise scheduling to treat channel concat as a virtual multi-source producer for BN/ReLU epilogues."""

import torch
import triton
import triton.language as tl

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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
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
def _invstd_kernel(
    var_ptr,
    invstd_ptr,
    C_TOTAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.arange(0, BLOCK_C)
    mask = c < C_TOTAL
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    denom = tl.sqrt(_f32_add(var, 1.0e-5))
    invstd = _f32_mul(_f32_div(1.0, denom), 1.0)
    tl.store(invstd_ptr + c, invstd, mask=mask)


@triton.jit
def _cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    x4_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C0: tl.constexpr,
    C_TOTAL: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    spatial_mask = spatial < HW
    mask = row_mask[:, None] & spatial_mask[None, :]

    n = rows // C_TOTAL
    c = rows - n * C_TOTAL
    rel = c - C0
    in0 = c < C0
    in1 = (rel >= 0) & (rel < 32)
    in2 = (rel >= 32) & (rel < 64)
    in3 = (rel >= 64) & (rel < 96)
    in4 = rel >= 96

    c0 = tl.where(in0, c, 0)
    c1 = tl.where(in1, rel, 0)
    c2 = tl.where(in2, rel - 32, 0)
    c3 = tl.where(in3, rel - 64, 0)
    c4 = tl.where(in4, rel - 96, 0)

    x0_offsets = (n[:, None] * C0 + c0[:, None]) * HW + spatial[None, :]
    xb_base = (n[:, None] * 32) * HW + spatial[None, :]
    x1_offsets = xb_base + c1[:, None] * HW
    x2_offsets = xb_base + c2[:, None] * HW
    x3_offsets = xb_base + c3[:, None] * HW
    x4_offsets = xb_base + c4[:, None] * HW

    x = tl.load(x0_ptr + x0_offsets, mask=mask & in0[:, None], other=0.0).to(tl.float32)
    x = _f32_add(x, tl.load(x1_ptr + x1_offsets, mask=mask & in1[:, None], other=0.0).to(tl.float32))
    x = _f32_add(x, tl.load(x2_ptr + x2_offsets, mask=mask & in2[:, None], other=0.0).to(tl.float32))
    x = _f32_add(x, tl.load(x3_ptr + x3_offsets, mask=mask & in3[:, None], other=0.0).to(tl.float32))
    x = _f32_add(x, tl.load(x4_ptr + x4_offsets, mask=mask & in4[:, None], other=0.0).to(tl.float32))

    mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

    y = _f32_mul(_f32_sub(x, mean[:, None]), invstd[:, None])
    y = _f32_add(_f32_mul(y, weight[:, None]), bias[:, None])
    y_bf16 = _round_bf16_to_f32(y)
    relu = tl.where(y_bf16 < 0.0, 0.0, y_bf16)

    out_offsets = rows[:, None] * HW + spatial[None, :]
    tl.store(out_ptr + out_offsets, relu, mask=mask)


def _forward(inputs, *, C0: int, H: int, BLOCK_ROWS: int, BLOCK_HW: int):
    x0, x1, x2, x3, x4, mean, var, weight, bias = inputs
    n = int(x0.shape[0])
    hw = H * H
    c_total = C0 + 128
    total_rows = n * c_total
    out = torch.empty_strided(
        (n, c_total, H, H),
        (c_total * hw, hw, H, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    invstd = torch.empty_strided((c_total,), (1,), device=x0.device, dtype=torch.float32)

    _invstd_kernel[(1,)](
        var,
        invstd,
        C_TOTAL=c_total,
        BLOCK_C=triton.next_power_of_2(c_total),
        num_warps=8,
        num_stages=1,
    )
    _cat_bn_relu_kernel[(triton.cdiv(total_rows, BLOCK_ROWS), triton.cdiv(hw, BLOCK_HW))](
        x0,
        x1,
        x2,
        x3,
        x4,
        mean,
        invstd,
        weight,
        bias,
        out,
        C0=C0,
        C_TOTAL=c_total,
        HW=hw,
        TOTAL_ROWS=total_rows,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="dbd3312e", C0=512, H=7, BLOCK_ROWS=16, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="9ae6b248", C0=256, H=14, BLOCK_ROWS=16, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="bc847cd3", C0=128, H=28, BLOCK_ROWS=16, BLOCK_HW=128)
@oracle_impl(hardware="B200", point="65d9b82c", C0=64, H=56, BLOCK_ROWS=16, BLOCK_HW=128)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
