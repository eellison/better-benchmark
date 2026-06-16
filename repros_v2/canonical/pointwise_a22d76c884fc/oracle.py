"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet seven-source virtual channel concat, fp32 BN-inference affine from bf16 parameters, explicit bf16 cast, and ReLU output for all four captured shapes in one Triton pass, whereas Inductor materializes the fixed channel cat before the downstream affine/ReLU pointwise work; Inductor cannot do this today because its scheduler does not keep multi-input channel concat as a virtual producer across the channel-dependent BN affine consumer and bf16 epilogue; the fix is SCHEDULER_FUSION: inline fixed channel concat producers into pointwise consumers with guarded source selection and preserved cast boundaries."""

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
    x3_ptr,
    x4_ptr,
    x5_ptr,
    x6_ptr,
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
    use2 = (tail >= 32) & (tail < 64)
    use3 = (tail >= 64) & (tail < 96)
    use4 = (tail >= 96) & (tail < 128)
    use5 = (tail >= 128) & (tail < 160)
    use6 = tail >= 160
    c1 = tl.where(use1, tail, 0)
    c2 = tl.where(use2, tail - 32, 0)
    c3 = tl.where(use3, tail - 64, 0)
    c4 = tl.where(use4, tail - 96, 0)
    c5 = tl.where(use5, tail - 128, 0)
    c6 = tl.where(use6, tail - 160, 0)

    x0_offsets = (n * C0 + c[:, None]) * HW + hw[None, :]
    x1_offsets = (n * 32 + c1[:, None]) * HW + hw[None, :]
    x2_offsets = (n * 32 + c2[:, None]) * HW + hw[None, :]
    x3_offsets = (n * 32 + c3[:, None]) * HW + hw[None, :]
    x4_offsets = (n * 32 + c4[:, None]) * HW + hw[None, :]
    x5_offsets = (n * 32 + c5[:, None]) * HW + hw[None, :]
    x6_offsets = (n * 32 + c6[:, None]) * HW + hw[None, :]

    x = tl.load(x0_ptr + x0_offsets, mask=mask & use0[:, None], other=0.0).to(tl.float32)
    x += tl.load(x1_ptr + x1_offsets, mask=mask & use1[:, None], other=0.0).to(tl.float32)
    x += tl.load(x2_ptr + x2_offsets, mask=mask & use2[:, None], other=0.0).to(tl.float32)
    x += tl.load(x3_ptr + x3_offsets, mask=mask & use3[:, None], other=0.0).to(tl.float32)
    x += tl.load(x4_ptr + x4_offsets, mask=mask & use4[:, None], other=0.0).to(tl.float32)
    x += tl.load(x5_ptr + x5_offsets, mask=mask & use5[:, None], other=0.0).to(tl.float32)
    x += tl.load(x6_ptr + x6_offsets, mask=mask & use6[:, None], other=0.0).to(tl.float32)

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


@oracle_impl(hardware="B200", point="60be6835", C0=512, C_OUT=704, H=7, W=7, BLOCK_C=8, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="6447e2c6", C0=256, C_OUT=448, H=14, W=14, BLOCK_C=8, BLOCK_HW=128, num_warps=4)
@oracle_impl(hardware="B200", point="20eb3168", C0=128, C_OUT=320, H=28, W=28, BLOCK_C=8, BLOCK_HW=256, num_warps=8)
@oracle_impl(hardware="B200", point="9869bb14", C0=64, C_OUT=256, H=56, W=56, BLOCK_C=8, BLOCK_HW=256, num_warps=8)
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
    x0, x1, x2, x3, x4, x5, x6, mean, var, weight, bias = inputs
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
        x3,
        x4,
        x5,
        x6,
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
