"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full RepVGG dual BN-inference affine, explicit bf16 branch materialization, bf16 branch add, ReLU, and 7x7 spatial mean directly into the returned `[128,1408]` view, whereas Inductor lowers the decomposed sub/sqrt/reciprocal/mul/add/cast/add/relu/mean graph through a generic fused reduction schedule over channels-last physical strides; Inductor cannot do this today because its norm canonicalizer does not recognize sibling BN-affine branches with required bf16 cast boundaries feeding a spatial mean; the fix is ALGEBRAIC_ELIMINATION: canonicalize inference BN branches into one reduction kernel while preserving the exact sqrt/reciprocal order, casts, strides, and output scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _dual_bn_relu_mean_kernel(
    mean0_ptr,
    x0_ptr,
    var0_ptr,
    weight0_ptr,
    bias0_ptr,
    mean1_ptr,
    x1_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    EPS_: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = rows - (rows // 1408) * 1408
    batches = rows // 1408
    hw = tl.arange(0, BLOCK_HW)
    h = hw // 7
    w = hw - h * 7

    valid_rows = rows < 180224
    valid_hw = hw < 49
    valid = valid_rows[:, None] & valid_hw[None, :]
    offsets = (
        batches[:, None] * 68992
        + channels[:, None]
        + h[None, :] * 9856
        + w[None, :] * 1408
    )

    x0 = tl.load(x0_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    mean0 = tl.load(mean0_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    var0 = tl.load(var0_ptr + channels, mask=valid_rows, other=1.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + channels, mask=valid_rows, other=1.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + channels, mask=valid_rows, other=0.0).to(tl.float32)

    invstd0 = 1.0 / tl.sqrt(var0 + EPS_)
    invstd1 = 1.0 / tl.sqrt(var1 + EPS_)
    branch0 = (x0 - mean0[:, None]) * invstd0[:, None]
    branch0 = branch0 * weight0[:, None] + bias0[:, None]
    branch1 = (x1 - mean1[:, None]) * invstd1[:, None]
    branch1 = branch1 * weight1[:, None] + bias1[:, None]

    branch0 = _round_to_bf16_f32(branch0)
    branch1 = _round_to_bf16_f32(branch1)
    summed = _round_to_bf16_f32(branch0 + branch1)
    relu = tl.where((summed > 0.0) | (summed != summed), summed, 0.0)
    reduced = tl.sum(tl.where(valid, relu, 0.0), axis=1) * (1.0 / 49.0)
    tl.store(out_ptr + rows, reduced, mask=valid_rows)


# 84ddc6ab: dual bf16 BN-affine branches over T([128,1408,7,7], stride=(68992,1,9856,1408)).
@oracle_impl(hardware="B200", point="84ddc6ab", BLOCK_ROWS=8, num_warps=1)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1, shape0, stride0, shape1 = inputs
    out = torch.empty_strided(
        (int(shape1[0]), int(shape1[1])),
        (int(shape1[1]), 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    _dual_bn_relu_mean_kernel[(triton.cdiv(180224, BLOCK_ROWS),)](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        out,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=64,
        EPS_=EPS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
