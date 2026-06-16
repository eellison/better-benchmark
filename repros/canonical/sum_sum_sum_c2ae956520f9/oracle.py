"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 avg-pool-backward exact-GELU-gradient return scope, including the channels-last dense bf16 gradient, returned bf16 sigmoid gate, f32 scalar sum over the bf16 gated producer, returned dense bf16 scalar-gradient tensor, bf16 sigmoid-gradient side tensor, and final f32 channel reduction from the same shared producer, whereas Inductor lowers the pool-backward/GELU producer, dense side stores, spatial reduction, scalar reduction, and dependent channel finalizer as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler cannot keep a fixed 2x2 pool-backward producer and multiple sibling reductions with returned side tensors resident in one cast-sensitive fusion plan; the fix is SCHEDULER_FUSION: teach reduction scheduling to form one multi-output plan that sinks the pool-backward stencil, exact libdevice erf/exp epilogue, bf16 materializations, and scalar/channel finalizers into the shared producer."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

from oracle_harness import oracle_impl


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _pointwise_kernel(
    low_ptr,
    add_ptr,
    gelu_ptr,
    gate_ptr,
    rhs_ptr,
    scalar_ptr,
    out_grad_ptr,
    out_scaled_ptr,
    scalar_partials_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c = offsets % C
    w = (offsets // C) % W
    h = (offsets // (C * W)) % H
    n = offsets // (C * H * W)
    low_offsets = n * C * PH * PW + ((h // 2) * PW + (w // 2)) * C + c

    pool_grad = tl.load(low_ptr + low_offsets, mask=mask, other=0.0).to(tl.float32)
    add_in = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gelu_x = tl.load(gelu_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    scalar = _bf16(tl.load(scalar_ptr).to(tl.float32))

    pool_back_compiled = pool_grad / 4.0
    base_compiled = (add_in + pool_back_compiled) * 0.9622504486493761
    base_compiled = base_compiled * 1.7015043497085571

    add = _bf16(add_in + pool_back_compiled)
    mul = _bf16(add * 0.9622504486493761)
    base = _bf16(mul * 1.7015043497085571)

    erf_val = libdevice.erf(gelu_x * 0.7071067811865476)
    cdf = (erf_val + 1.0) * 0.5
    pdf = tl_math.exp((gelu_x * gelu_x) * -0.5) * 0.3989422804014327
    gelu_grad = cdf + gelu_x * pdf

    grad_for_scalar = (base_compiled * gelu_grad).to(tl.bfloat16).to(tl.float32)
    grad = _bf16(base * gelu_grad)
    grad_bf16 = grad.to(tl.bfloat16)
    tl.store(out_grad_ptr + offsets, grad_bf16, mask=mask)

    mul9 = _bf16(grad * 0.2)
    scaled = _bf16(mul9 * scalar)
    scaled = _bf16(scaled * 2.0)
    tl.store(out_scaled_ptr + offsets, scaled.to(tl.bfloat16), mask=mask)

    sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    scalar_terms = (grad_for_scalar * 0.2) * ((rhs * sigmoid) * 2.0)
    tl.store(scalar_partials_ptr + tl.program_id(0), tl.sum(tl.where(mask, scalar_terms, 0.0), axis=0))


@triton.jit
def _side_reduce_kernel(
    out_scaled_ptr,
    rhs_ptr,
    gate_ptr,
    out_sigmoid_ptr,
    out_side_ptr,
    side_nc_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    cs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = cs < C

    nc_offsets = n * C + cs
    sigmoid_bf16 = tl.sigmoid(tl.load(gate_ptr + nc_offsets, mask=c_mask, other=0.0).to(tl.float32)).to(tl.bfloat16)
    sigmoid = sigmoid_bf16.to(tl.float32)
    tl.store(out_sigmoid_ptr + nc_offsets, sigmoid_bf16, mask=c_mask)

    hw_offsets = tl.arange(0, BLOCK_HW)
    sum2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for hw_base in tl.static_range(0, HW, BLOCK_HW):
        hw = hw_base + hw_offsets
        mask = c_mask[:, None] & (hw[None, :] < HW)
        offsets = n * C * HW + hw[None, :] * C + cs[:, None]
        scaled = tl.load(out_scaled_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum2 += tl.sum(tl.where(mask, _bf16(scaled * rhs), 0.0), axis=1)
    side = _bf16(sum2) * (sigmoid * (1.0 - sigmoid))
    side_bf16 = side.to(tl.bfloat16)
    tl.store(out_side_ptr + nc_offsets, side_bf16, mask=c_mask)
    tl.store(side_nc_ptr + nc_offsets, side_bf16.to(tl.float32), mask=c_mask)


@triton.jit
def _finalize_channels_kernel(
    side_nc_ptr,
    out_channel_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c_block = tl.program_id(0)
    ns = tl.arange(0, BLOCK_N)
    cs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (ns[:, None] < N) & (cs[None, :] < C)
    offsets = ns[:, None] * C + cs[None, :]
    side_vals = tl.load(side_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    channel = _bf16(tl.sum(side_vals, axis=0))
    tl.store(out_channel_ptr + cs, channel, mask=cs < C)


@triton.jit
def _finalize_scalar_kernel(
    scalar_blocks_ptr,
    out_scalar_ptr,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCKS)
    mask = offsets < NUM_BLOCKS
    vals = tl.load(scalar_blocks_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_scalar_ptr, tl.sum(vals, axis=0))


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _launch(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    PH: int,
    PW: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    POINT_BLOCK: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0_1, _arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    n = int(arg0_1.shape[0])
    hw = H * W
    device = arg0_1.device

    dense_shape = (n, C, H, W)
    dense_stride = (C * hw, 1, W * C, C)
    gate_shape = (n, C, 1, 1)
    gate_stride = (C, 1, 1, 1)

    out_grad = torch.empty_strided(dense_shape, dense_stride, device=device, dtype=torch.bfloat16)
    out_sigmoid = torch.empty_strided(gate_shape, gate_stride, device=device, dtype=torch.bfloat16)
    out_scalar = torch.empty_strided((), (), device=device, dtype=torch.float32)
    out_scaled = torch.empty_strided(dense_shape, dense_stride, device=device, dtype=torch.bfloat16)
    out_side = torch.empty_strided(gate_shape, gate_stride, device=device, dtype=torch.bfloat16)
    out_channel = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    side_nc = torch.empty_strided((n, C), (C, 1), device=device, dtype=torch.float32)
    total = n * C * hw
    scalar_blocks_count = triton.cdiv(total, POINT_BLOCK)
    scalar_blocks = torch.empty_strided((scalar_blocks_count,), (1,), device=device, dtype=torch.float32)

    _pointwise_kernel[(scalar_blocks_count,)](
        arg0_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out_grad,
        out_scaled,
        scalar_blocks,
        C=C,
        H=H,
        W=W,
        PH=PH,
        PW=PW,
        TOTAL=total,
        BLOCK=POINT_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    _side_reduce_kernel[(n, triton.cdiv(C, BLOCK_C))](
        out_scaled,
        arg5_1,
        arg4_1,
        out_sigmoid,
        out_side,
        side_nc,
        C=C,
        HW=hw,
        N=n,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )

    final_blocks = triton.cdiv(C, FINAL_BLOCK_C)
    _finalize_channels_kernel[(final_blocks,)](
        side_nc,
        out_channel,
        C=C,
        N=n,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_N=_ceil_pow2(n),
        num_warps=8,
        num_stages=3,
    )
    _finalize_scalar_kernel[(1,)](
        scalar_blocks,
        out_scalar,
        NUM_BLOCKS=scalar_blocks_count,
        BLOCKS=_ceil_pow2(scalar_blocks_count),
        num_warps=4,
        num_stages=3,
    )
    return out_grad, out_sigmoid, out_scalar, out_scaled, out_side, out_channel


@oracle_impl(hardware="B200", point="1a7f832d", C=512, H=24, W=24, PH=12, PW=12, BLOCK_C=16, BLOCK_HW=256, POINT_BLOCK=4096, FINAL_BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="cf12eab2", C=1536, H=12, W=12, PH=6, PW=6, BLOCK_C=64, BLOCK_HW=32, POINT_BLOCK=512, FINAL_BLOCK_C=16, num_warps=4)
def oracle_forward(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    PH: int,
    PW: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    POINT_BLOCK: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    return _launch(
        inputs,
        C=C,
        H=H,
        W=W,
        PH=PH,
        PW=PW,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        POINT_BLOCK=POINT_BLOCK,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps,
    )
