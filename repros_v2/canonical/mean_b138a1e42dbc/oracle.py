"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 ResNet inference BN-affine, explicit bf16 round, residual bf16 add, NaN-preserving ReLU, and spatial mean directly into the returned `[N,C]` view; whereas Inductor lowers the decomposed normalization and mean as a generic fused reduction that repeats batch-invariant BN algebra inside each spatial row; Inductor cannot do this today because its reduction simplifier does not hoist inference-BN scale/shift through activation-fed spatial means while preserving bf16 cast boundaries; the fix is ALGEBRAIC_ELIMINATION: canonicalize BN-inference affine feeding spatial mean reductions into reusable per-channel scale/shift with explicit bf16 round/add semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5
BLOCK_HW = 64


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
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_residual_relu_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW_: tl.constexpr,
    EPS_: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channel_offsets = row_offsets - (row_offsets // CHANNELS) * CHANNELS
    hw_offsets = tl.arange(0, BLOCK_HW_)

    valid_rows = row_offsets < TOTAL_ROWS
    valid_hw = hw_offsets < HW
    valid = valid_rows[:, None] & valid_hw[None, :]

    element_offsets = row_offsets[:, None] * HW + hw_offsets[None, :]
    x = tl.load(x_ptr + element_offsets, mask=valid, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + element_offsets, mask=valid, other=0.0).to(
        tl.float32
    )

    mean = tl.load(mean_ptr + channel_offsets, mask=valid_rows, other=0.0).to(
        tl.float32
    )
    var = tl.load(var_ptr + channel_offsets, mask=valid_rows, other=1.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + channel_offsets, mask=valid_rows, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + channel_offsets, mask=valid_rows, other=0.0).to(
        tl.float32
    )

    invstd = 1.0 / tl.sqrt(var + EPS_)
    y = (x - mean[:, None]) * invstd[:, None]
    y = y * weight[:, None] + bias[:, None]
    y = _round_bf16_to_f32(y)
    y = _round_bf16_to_f32(y + residual)
    y = _relu_preserve_nan(y)
    reduced = tl.sum(tl.where(valid, y, 0.0), axis=1) * (1.0 / HW)
    tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


# 187a2d53: (T([512], bf16), T([8,512,7,7], bf16), ...)
@oracle_impl(hardware="B200", point="187a2d53", BLOCK_ROWS=8, num_warps=4)
# 08581e62: (T([2048], bf16), T([32,2048,7,7], bf16), ...)
@oracle_impl(hardware="B200", point="08581e62", BLOCK_ROWS=8, num_warps=4)
# 7111f2e4: (T([2048], bf16), T([8,2048,7,7], bf16), ...)
@oracle_impl(hardware="B200", point="7111f2e4", BLOCK_ROWS=8, num_warps=4)
# e2161da4: (T([64], bf16), T([128,64,8,8], bf16), ...)
@oracle_impl(hardware="B200", point="e2161da4", BLOCK_ROWS=8, num_warps=4)
# 2f5dac5e: (T([2048], bf16), T([128,2048,7,7], bf16), ...)
@oracle_impl(hardware="B200", point="2f5dac5e", BLOCK_ROWS=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    mean, x, var, weight, bias, residual, view_shape = inputs
    batch = int(view_shape[0])
    channels = int(view_shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total_rows = batch * channels
    output = torch.empty_strided(
        (batch, channels), (channels, 1), device=x.device, dtype=torch.bfloat16
    )
    _bn_residual_relu_mean_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        output,
        TOTAL_ROWS=total_rows,
        CHANNELS=channels,
        HW=hw,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        EPS_=EPS,
        num_warps=num_warps,
        num_stages=3,
    )
    return output
