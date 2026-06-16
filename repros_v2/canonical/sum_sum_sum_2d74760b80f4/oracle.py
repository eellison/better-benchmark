"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 gated GELU-backward scope, including the returned sigmoid, add_3, mul_16, sigmoid-gradient side tensor, scalar sum, and both channel sums from one shared producer; whereas Inductor schedules the dense bf16 producer, spatial reduction epilogue, scalar reduction, and channel reductions as separate generic regions; Inductor cannot do this today because its scheduler cannot keep sibling reductions with dense side outputs and a dependent per-(N,C) bf16 epilogue resident in one fusion group; the fix is SCHEDULER_FUSION: extend multi-output reduction fusion to preserve bf16 cast boundaries while emitting shared dense stores plus scalar and channel finalizers."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

from oracle_harness import oracle_impl


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _spatial_kernel(
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    out_sigmoid,
    out_add3,
    out_mul16,
    out_gate,
    scalar_nc,
    gate_nc,
    add3_nc,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    cs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = cs < C

    gate_arg = tl.load(arg1 + n * C + cs, mask=c_mask, other=0.0).to(tl.float32)
    sigmoid_bf = tl.sigmoid(gate_arg).to(tl.bfloat16)
    sigmoid_f = sigmoid_bf.to(tl.float32)
    scalar = tl.load(arg3).to(tl.float32)
    scalar_bf = _bf16(scalar)

    tl.store(out_sigmoid + n * C + cs, sigmoid_bf, mask=c_mask)

    acc_scalar = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_add3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    hw_offsets = tl.arange(0, BLOCK_HW)

    for hw_base in tl.static_range(0, HW, BLOCK_HW):
        hw = hw_base + hw_offsets
        hw_mask = hw < HW
        dense_offsets = n * C * HW + hw[None, :] * C + cs[:, None]
        mask = c_mask[:, None] & hw_mask[None, :]

        a0 = tl.load(arg0 + dense_offsets, mask=mask, other=0.0).to(tl.float32)
        a2 = tl.load(arg2 + dense_offsets, mask=mask, other=0.0).to(tl.float32)
        a4 = tl.load(arg4 + dense_offsets, mask=mask, other=0.0).to(tl.float32)
        a5 = tl.load(arg5 + dense_offsets, mask=mask, other=0.0).to(tl.float32)

        mul = _bf16(a0 * 0.9805806756909201)
        mul_1 = _bf16(mul * 1.7015043497085571)

        mul_2 = _bf16(a2 * sigmoid_f[:, None])
        mul_3 = _bf16(mul_2 * 2.0)
        mul_4 = _bf16(mul_3 * scalar_bf)
        mul_5 = _bf16(mul_4 * 0.2)
        add = _bf16(mul_5 + a4)

        mul_6 = add * 0.7071067811865476
        erf = libdevice.erf(mul_6)
        add_1 = erf + 1.0
        mul_7 = add_1 * 0.5
        mul_8 = add * add
        mul_9 = mul_8 * -0.5
        exp = tl_math.exp(mul_9)
        mul_10 = exp * 0.3989422804014327
        mul_11 = add * mul_10
        add_2 = mul_7 + mul_11

        mul_12 = mul_1 * add_2
        convert_2 = _bf16(mul_12)
        add_3 = _bf16(a5 + convert_2)
        mul_13 = _bf16(add_3 * 0.2)
        mul_14 = _bf16(mul_13 * mul_3)
        mul_15 = _bf16(mul_13 * scalar_bf)
        mul_16 = _bf16(mul_15 * 2.0)
        mul_17 = _bf16(mul_16 * a2)

        tl.store(out_add3 + dense_offsets, add_3.to(tl.bfloat16), mask=mask)
        tl.store(out_mul16 + dense_offsets, mul_16.to(tl.bfloat16), mask=mask)

        acc_scalar += tl.sum(tl.where(mask, mul_14, 0.0), axis=1)
        acc_gate += tl.sum(tl.where(mask, mul_17, 0.0), axis=1)
        acc_add3 += tl.sum(tl.where(mask, add_3, 0.0), axis=1)

    sum_2_bf = _bf16(acc_gate)
    sigmoid_grad = sigmoid_f * (1.0 - sigmoid_f)
    gate_bf = _bf16(sum_2_bf * sigmoid_grad)
    nc_offsets = n * C + cs

    tl.store(out_gate + nc_offsets, gate_bf.to(tl.bfloat16), mask=c_mask)
    tl.store(scalar_nc + nc_offsets, acc_scalar, mask=c_mask)
    tl.store(gate_nc + nc_offsets, gate_bf, mask=c_mask)
    tl.store(add3_nc + nc_offsets, acc_add3, mask=c_mask)


@triton.jit
def _finalize_channels_kernel(
    scalar_nc,
    gate_nc,
    add3_nc,
    scalar_blocks,
    out_sum3,
    out_sum4,
    C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    ns = tl.arange(0, BLOCK_N)
    cs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cs[None, :] < C
    offsets = ns[:, None] * C + cs[None, :]

    scalar_vals = tl.load(scalar_nc + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_vals = tl.load(gate_nc + offsets, mask=mask, other=0.0).to(tl.float32)
    add3_vals = tl.load(add3_nc + offsets, mask=mask, other=0.0).to(tl.float32)

    scalar_per_c = tl.sum(scalar_vals, axis=0)
    gate_sum = tl.sum(gate_vals, axis=0)
    add3_sum = tl.sum(add3_vals, axis=0)
    c_mask = cs < C

    tl.store(out_sum3 + cs, _bf16(gate_sum), mask=c_mask)
    tl.store(out_sum4 + cs, _bf16(add3_sum), mask=c_mask)
    tl.store(scalar_blocks + c_block, tl.sum(tl.where(c_mask, scalar_per_c, 0.0), axis=0))


@triton.jit
def _finalize_scalar_kernel(
    scalar_blocks,
    out_scalar,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    offs = tl.arange(0, BLOCKS)
    mask = offs < NUM_BLOCKS
    vals = tl.load(scalar_blocks + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_scalar, tl.sum(vals, axis=0))


def _run(inputs, C, H, W, BLOCK_C, BLOCK_HW, FINAL_BLOCK_C, num_warps):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    n = arg0.shape[0]
    hw = H * W
    device = arg0.device

    out_sigmoid = torch.empty((n, C, 1, 1), device=device, dtype=torch.bfloat16)
    out_add3 = torch.empty_strided((n, C, H, W), (C * hw, 1, W * C, C), device=device, dtype=torch.bfloat16)
    out_scalar = torch.empty((), device=device, dtype=torch.float32)
    out_mul16 = torch.empty_strided((n, C, H, W), (C * hw, 1, W * C, C), device=device, dtype=torch.bfloat16)
    out_gate = torch.empty((n, C, 1, 1), device=device, dtype=torch.bfloat16)
    out_sum3 = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((C,), device=device, dtype=torch.float32)
    scalar_nc = torch.empty((n, C), device=device, dtype=torch.float32)
    gate_nc = torch.empty((n, C), device=device, dtype=torch.float32)
    add3_nc = torch.empty((n, C), device=device, dtype=torch.float32)

    grid = (n, triton.cdiv(C, BLOCK_C))
    _spatial_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        out_sigmoid,
        out_add3,
        out_mul16,
        out_gate,
        scalar_nc,
        gate_nc,
        add3_nc,
        C,
        H,
        W,
        hw,
        BLOCK_C,
        BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )

    num_c_blocks = triton.cdiv(C, FINAL_BLOCK_C)
    scalar_blocks = torch.empty((num_c_blocks,), device=device, dtype=torch.float32)
    _finalize_channels_kernel[(num_c_blocks,)](
        scalar_nc,
        gate_nc,
        add3_nc,
        scalar_blocks,
        out_sum3,
        out_sum4,
        C,
        n,
        FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    block_pow2 = triton.next_power_of_2(num_c_blocks)
    _finalize_scalar_kernel[(1,)](
        scalar_blocks,
        out_scalar,
        num_c_blocks,
        block_pow2,
        num_warps=1,
        num_stages=3,
    )
    return out_sigmoid, out_add3, out_scalar, out_mul16, out_gate, out_sum3, out_sum4


@oracle_impl(hardware="B200", point="adbc9760", C=512, H=24, W=24, BLOCK_C=32, BLOCK_HW=256, FINAL_BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="bfd27ee1", C=1536, H=12, W=12, BLOCK_C=8, BLOCK_HW=128, FINAL_BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="9d70a1e6", C=1536, H=6, W=6, BLOCK_C=16, BLOCK_HW=64, FINAL_BLOCK_C=16, num_warps=4)
def oracle_forward(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    return _run(inputs, C, H, W, BLOCK_C, BLOCK_HW, FINAL_BLOCK_C, num_warps)
