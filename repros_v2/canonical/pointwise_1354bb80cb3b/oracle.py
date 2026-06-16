"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet virtual channel concat, fp32 BN-inference affine from bf16 parameters, explicit bf16 cast, and ReLU output for all four captured shapes in one Triton pass; whereas Inductor materializes the fixed channel cat before the downstream affine/ReLU pointwise work; Inductor cannot do this today because its scheduler does not keep multi-input channel concat as a virtual producer across the channel-dependent BN affine consumer and bf16 epilogue; the fix is SCHEDULER_FUSION: inline fixed channel concat producers into pointwise consumers with guarded source selection and preserved cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C0: tl.constexpr,
    C_OUT: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid_c = c < C_OUT
    valid_hw = hw < HW
    mask = valid_c[:, None] & valid_hw[None, :]

    tail = c - C0
    use0 = c < C0
    use1 = (tail >= 0) & (tail < 32)
    use2 = tail >= 32
    c1 = tl.where(use1, tail, 0)
    c2 = tl.where(use2, tail - 32, 0)

    x0_offsets = (n * C0 + c[:, None]) * HW + hw[None, :]
    x1_offsets = (n * 32 + c1[:, None]) * HW + hw[None, :]
    x2_offsets = (n * 32 + c2[:, None]) * HW + hw[None, :]

    x = tl.load(x0_ptr + x0_offsets, mask=mask & use0[:, None], other=0.0).to(tl.float32)
    x += tl.load(x1_ptr + x1_offsets, mask=mask & use1[:, None], other=0.0).to(tl.float32)
    x += tl.load(x2_ptr + x2_offsets, mask=mask & use2[:, None], other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + c, mask=valid_c, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=valid_c, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=valid_c, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=valid_c, other=0.0).to(tl.float32)

    invstd = 1.0 / tl.sqrt(var + 1.0e-5)
    y = (x - mean[:, None]) * invstd[:, None]
    y = y * weight[:, None] + bias[:, None]
    y_bf16 = _round_bf16_to_f32(y)
    relu = tl.where(y_bf16 < 0.0, 0.0, y_bf16)

    out_offsets = (n * C_OUT + c[:, None]) * HW + hw[None, :]
    tl.store(out_ptr + out_offsets, relu, mask=mask)


@oracle_impl(hardware="B200", point="c807f0f8", C0=512, C_OUT=576, H=7, W=7, BLOCK_C=8, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="7dd3ded5", C0=256, C_OUT=320, H=14, W=14, BLOCK_C=8, BLOCK_HW=128, num_warps=4)
@oracle_impl(hardware="B200", point="5f7d3f19", C0=128, C_OUT=192, H=28, W=28, BLOCK_C=8, BLOCK_HW=256, num_warps=8)
@oracle_impl(hardware="B200", point="d3e900e3", C0=64, C_OUT=128, H=56, W=56, BLOCK_C=8, BLOCK_HW=256, num_warps=8)
def oracle_forward(
    inputs,
    *,
    C0: int,
    C_OUT: int,
    H: int,
    W: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    num_warps: int,
):
    x0, x1, x2, mean, var, weight, bias = inputs
    batch = x0.shape[0]
    hw = H * W
    out = torch.empty_strided(
        (batch, C_OUT, H, W),
        (C_OUT * hw, hw, W, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    grid = (batch, triton.cdiv(C_OUT, BLOCK_C), triton.cdiv(hw, BLOCK_HW))
    _cat_bn_relu_kernel[grid](
        x0,
        x1,
        x2,
        mean,
        var,
        weight,
        bias,
        out,
        C0=C0,
        C_OUT=C_OUT,
        HW=hw,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
    )
    return out
