"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 BN-inference affine, explicit bf16 round-trip, hard-swish, second bf16 round-trip, and 7x7 spatial mean directly into the returned `[32,960]` view, whereas Inductor lowers the decomposed sub/sqrt/reciprocal/affine/cast/add/minmax/mul/div/cast/mean graph through its generic fused reduction schedule; Inductor cannot do this today because the norm lowering lacks a guarded BN-affine plus hard-swish spatial-mean plan that preserves the observable bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a benchmarked BN-inference activation-spatial-mean lowering that keeps per-channel parameters and hard-swish inside one reduction kernel with exact dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 0.001


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_hardswish_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    EPS_: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = rows - (rows // 960) * 960
    hw = tl.arange(0, BLOCK_HW)

    valid_rows = rows < 30720
    valid_hw = hw < 49
    valid = valid_rows[:, None] & valid_hw[None, :]
    offsets = rows[:, None] * 49 + hw[None, :]

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=valid_rows, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var + EPS_)
    affine = (x - mean[:, None]) * invstd[:, None]
    affine = affine * weight[:, None] + bias[:, None]
    affine = _round_to_bf16_f32(affine)
    relu6 = affine + 3.0
    relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
    relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
    hardswish = affine * relu6 * (1.0 / 6.0)
    hardswish = _round_to_bf16_f32(hardswish)
    reduced = tl.sum(tl.where(valid, hardswish, 0.0), axis=1) * (1.0 / 49.0)
    tl.store(out_ptr + rows, reduced, mask=valid_rows)


# 2c1989e8: (T([960], bf16), T([32,960,7,7], bf16), T([960], bf16), ...)
@oracle_impl(hardware="B200", point="2c1989e8", BLOCK_ROWS=16, num_warps=1)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    mean, x, var, weight, bias, shape = inputs
    out = torch.empty_strided(
        (int(shape[0]), int(shape[1])),
        (int(shape[1]), 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bn_hardswish_mean_kernel[(triton.cdiv(30720, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=64,
        EPS_=EPS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
