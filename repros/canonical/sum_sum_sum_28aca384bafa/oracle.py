"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 gated exact-GELU backward scope, including the channels-last dense add output, returned bf16 sigmoid gate, f32 scalar sum over the bf16 gated producer, returned bf16 dense scalar-gradient output, bf16-rounded spatial reduction feeding the sigmoid-gradient side tensor, and final bf16 channel sum converted back to f32, whereas Inductor lowers the shared GELU-gradient producer, dense stores, spatial reduction, scalar reduction, and dependent channel reduction as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler does not keep one bf16-cast-sensitive producer resident across reductions with different output ranks plus returned side tensors; the fix is SCHEDULER_FUSION: teach reduction scheduling to form one multi-output plan that sinks the required bf16 materializations and finalizers into the shared producer."""

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
def _producer_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    out_add_ptr,
    out_sigmoid_ptr,
    out_mul14_ptr,
    sum1_nc_ptr,
    sum2_nc_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_vec = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)

    gate = tl.load(arg3_ptr + n * C + c_vec, mask=c_vec < C, other=0.0).to(tl.float32)
    sigmoid_f32 = _f32_div(1.0, _f32_add(1.0, libdevice.exp(-gate)))
    sigmoid = _round_to_bf16_f32(sigmoid_f32)
    tl.store(out_sigmoid_ptr + n * C + c_vec, sigmoid, mask=c_vec < C)

    scalar = _round_to_bf16_f32(tl.load(arg5_ptr).to(tl.float32))
    sum1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    hw_vec = tl.arange(0, BLOCK_HW)
    c = c_vec[:, None]
    for hw_base in tl.static_range(0, HW, BLOCK_HW):
        hw = hw_base + hw_vec[None, :]
        active = (c < C) & (hw < HW)
        offsets = n * C * HW + hw * C + c

        arg0 = tl.load(arg0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        arg1 = tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        arg2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)

        mul = _round_to_bf16_f32(_f32_mul(arg0, 0.9622504486493761))
        mul_1 = _round_to_bf16_f32(_f32_mul(mul, 1.7015043497085571))

        erf_arg = _f32_mul(arg1, 0.7071067811865476)
        erf_val = libdevice.erf(erf_arg)
        mul_3 = _f32_mul(_f32_add(erf_val, 1.0), 0.5)
        mul_4 = _f32_mul(arg1, arg1)
        exp_arg = _f32_mul(mul_4, -0.5)
        exp_val = libdevice.exp(exp_arg)
        mul_6 = _f32_mul(exp_val, 0.3989422804014327)
        mul_7 = _f32_mul(arg1, mul_6)
        add_1 = _f32_add(mul_3, mul_7)
        convert_2 = _round_to_bf16_f32(_f32_mul(mul_1, add_1))

        add_2_f32 = _round_to_bf16_f32(_f32_add(arg2, convert_2))
        tl.store(out_add_ptr + offsets, add_2_f32, mask=active)

        arg4 = tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mul_9 = _round_to_bf16_f32(_f32_mul(add_2_f32, 0.2))
        mul_10 = _round_to_bf16_f32(_f32_mul(arg4, sigmoid[:, None]))
        mul_11 = _round_to_bf16_f32(_f32_mul(mul_10, 2.0))
        mul_12 = _round_to_bf16_f32(_f32_mul(mul_9, mul_11))

        mul_13 = _round_to_bf16_f32(_f32_mul(mul_9, scalar))
        mul_14_f32 = _round_to_bf16_f32(_f32_mul(mul_13, 2.0))
        tl.store(out_mul14_ptr + offsets, mul_14_f32, mask=active)

        mul_15 = _round_to_bf16_f32(_f32_mul(mul_14_f32, arg4))
        sum1 += tl.sum(tl.where(active, mul_12, 0.0), axis=1)
        sum2 += tl.sum(tl.where(active, mul_15, 0.0), axis=1)

    nc_offsets = n * C + c_vec
    tl.store(sum1_nc_ptr + nc_offsets, sum1, mask=c_vec < C)
    tl.store(sum2_nc_ptr + nc_offsets, sum2, mask=c_vec < C)


@triton.jit
def _side_kernel(
    sum2_nc_ptr,
    sigmoid_ptr,
    out_side_ptr,
    C: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    sum2 = tl.load(sum2_nc_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sigmoid = tl.load(sigmoid_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    rounded_sum2 = _round_to_bf16_f32(sum2)
    one_minus = _f32_sub(1.0, sigmoid)
    sigmoid_grad = _f32_mul(sigmoid, one_minus)
    side = _round_to_bf16_f32(_f32_mul(rounded_sum2, sigmoid_grad))
    tl.store(out_side_ptr + offsets, side, mask=active)


@triton.jit
def _final_channels_kernel(
    sum1_nc_ptr,
    out_side_ptr,
    scalar_blocks_ptr,
    out_channel_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block = tl.program_id(0)
    c = block * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    active = (n[:, None] < N) & (c[None, :] < C)
    offsets = n[:, None] * C + c[None, :]

    sum1_vals = tl.load(sum1_nc_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    side_vals = tl.load(out_side_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    scalar_per_channel = tl.sum(sum1_vals, axis=0)
    channel = _round_to_bf16_f32(tl.sum(side_vals, axis=0))

    c_mask = c < C
    tl.store(out_channel_ptr + c, channel, mask=c_mask)
    tl.store(scalar_blocks_ptr + block, tl.sum(tl.where(c_mask, scalar_per_channel, 0.0), axis=0))


@triton.jit
def _final_scalar_kernel(
    scalar_blocks_ptr,
    out_scalar_ptr,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCKS)
    active = offsets < NUM_BLOCKS
    vals = tl.load(scalar_blocks_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    tl.store(out_scalar_ptr, tl.sum(vals, axis=0))


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _launch(inputs, *, H, BLOCK_C, BLOCK_HW, SIDE_BLOCK, FINAL_BLOCK_C, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    hw = H * H
    device = arg0_1.device

    dense_stride = (c * hw, 1, H * c, c)
    gate_stride = (c, 1, 1, 1)
    dense_shape = (n, c, H, H)
    gate_shape = (n, c, 1, 1)

    out_add = torch.empty_strided(dense_shape, dense_stride, device=device, dtype=torch.bfloat16)
    out_sigmoid = torch.empty_strided(gate_shape, gate_stride, device=device, dtype=torch.bfloat16)
    out_scalar = torch.empty_strided((), (), device=device, dtype=torch.float32)
    out_mul14 = torch.empty_strided(dense_shape, dense_stride, device=device, dtype=torch.bfloat16)
    out_side = torch.empty_strided(gate_shape, gate_stride, device=device, dtype=torch.bfloat16)
    out_channel = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    sum1_nc = torch.empty_strided((n, c), (c, 1), device=device, dtype=torch.float32)
    sum2_nc = torch.empty_strided((n, c), (c, 1), device=device, dtype=torch.float32)
    final_blocks = triton.cdiv(c, FINAL_BLOCK_C)
    scalar_blocks = torch.empty_strided((final_blocks,), (1,), device=device, dtype=torch.float32)

    _producer_kernel[(n, triton.cdiv(c, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out_add,
        out_sigmoid,
        out_mul14,
        sum1_nc,
        sum2_nc,
        C=c,
        HW=hw,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    _side_kernel[(triton.cdiv(n * c, SIDE_BLOCK),)](
        sum2_nc,
        out_sigmoid,
        out_side,
        C=c,
        TOTAL=n * c,
        BLOCK=SIDE_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _final_channels_kernel[(final_blocks,)](
        sum1_nc,
        out_side,
        scalar_blocks,
        out_channel,
        C=c,
        N=n,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_N=_ceil_pow2(n),
        num_warps=8,
        num_stages=3,
    )
    _final_scalar_kernel[(1,)](
        scalar_blocks,
        out_scalar,
        NUM_BLOCKS=final_blocks,
        BLOCKS=_ceil_pow2(final_blocks),
        num_warps=4,
        num_stages=3,
    )
    return out_add, out_sigmoid, out_scalar, out_mul14, out_side, out_channel


@oracle_impl(hardware="B200", point="bfd27ee1", H=12, BLOCK_C=64, BLOCK_HW=32, SIDE_BLOCK=256, FINAL_BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="9d70a1e6", H=6, BLOCK_C=128, BLOCK_HW=64, SIDE_BLOCK=256, FINAL_BLOCK_C=16, num_warps=4)
def oracle_forward(inputs, *, H, BLOCK_C, BLOCK_HW, SIDE_BLOCK, FINAL_BLOCK_C, num_warps):
    return _launch(
        inputs,
        H=H,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        SIDE_BLOCK=SIDE_BLOCK,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps,
    )
