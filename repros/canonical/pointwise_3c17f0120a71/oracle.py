"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes both ShuffleNet bf16 BN-inference affine/ReLU branches and writes them directly into the final channel-shuffled backing tensor returned by the split views, whereas Inductor materializes the branch cat and then schedules a separate view/permute/clone/view/split layout chain; Inductor cannot do this today because scheduler fusion does not propagate the fixed channel-shuffle consumer layout and split offsets back into the two affine producers while preserving the explicit bf16 cast before ReLU; the fix is SCHEDULER_FUSION: keep the virtual cat plus channel-shuffle layout in codegen and sink the two BN/ReLU stores directly into the final split backing allocation."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

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
def _bn_relu_value(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, offsets, channels, mask):
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=mask, other=0.0).to(tl.float32)

    invstd = _f32_mul(1.0 / libdevice.sqrt(_f32_add(var, 1.0e-5)), 1.0)
    normalized = _f32_mul(_f32_sub(x, mean), invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    zero = tl.full(rounded.shape, 0.0, tl.float32).to(tl.bfloat16)
    return tl.where(rounded != rounded, rounded, tl.where(rounded < zero, zero, rounded))


@triton.jit
def _dual_bn_relu_shuffle_kernel(
    mean_a_ptr,
    x_a_ptr,
    var_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    mean_b_ptr,
    x_b_ptr,
    var_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    hw = offsets % HW
    c = (offsets // HW) % C
    n = offsets // (C * HW)
    src_offsets = n * C * HW + c * HW + hw

    a = _bn_relu_value(x_a_ptr, mean_a_ptr, var_a_ptr, weight_a_ptr, bias_a_ptr, src_offsets, c, mask)
    b = _bn_relu_value(x_b_ptr, mean_b_ptr, var_b_ptr, weight_b_ptr, bias_b_ptr, src_offsets, c, mask)

    out_offsets = n * OUT_C * HW + (2 * c) * HW + hw
    tl.store(out_ptr + out_offsets, a, mask=mask)
    tl.store(out_ptr + out_offsets + HW, b, mask=mask)


@oracle_impl(hardware="B200", point="f47c0655", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    mean_a, x_a, var_a, weight_a, bias_a, mean_b, x_b, var_b, weight_b, bias_b, _shape0, _shape1 = inputs
    shuffled = torch.empty_strided(
        (64, 232, 14, 14),
        (232 * 14 * 14, 14 * 14, 14, 1),
        device=x_a.device,
        dtype=torch.bfloat16,
    )
    _dual_bn_relu_shuffle_kernel[(triton.cdiv(64 * 116 * 14 * 14, BLOCK),)](
        mean_a,
        x_a,
        var_a,
        weight_a,
        bias_a,
        mean_b,
        x_b,
        var_b,
        weight_b,
        bias_b,
        shuffled,
        TOTAL=64 * 116 * 14 * 14,
        C=116,
        HW=14 * 14,
        OUT_C=232,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return shuffled[:, :116, :, :], shuffled[:, 116:, :, :]
