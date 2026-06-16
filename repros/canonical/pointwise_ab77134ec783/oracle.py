"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet singleton-gate residual add, SiLU scale, and 2x2 stride-2 avg_pool2d scope in one Triton output-tiled stencil kernel, storing the visible bf16 residual-add output, the visible bf16 scaled-SiLU output, and the pooled bf16 output while reusing the rounded scaled-SiLU values for the pool, whereas Inductor materializes the broadcast pointwise producer and then lowers the fixed-window pooling consumer separately; Inductor cannot do this today because scheduler fusion does not sink a returned broadcast-gated pointwise producer through avg_pool2d while preserving the side outputs and bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach fixed-window pooling codegen to fuse broadcast pointwise producer DAGs into a multi-output stencil plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


FINAL_SCALE = 0.8980265101338745


@triton.jit
def _producer_values(gate_bf16, x, residual, FINAL_SCALE_: tl.constexpr):
    mul0 = (x * gate_bf16.to(tl.float32)).to(tl.bfloat16)
    mul1 = (mul0.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul2 = (mul1.to(tl.float32) * 0.2).to(tl.bfloat16)
    add_value = (mul2.to(tl.float32) + residual).to(tl.bfloat16)
    add_f32 = add_value.to(tl.float32)
    silu = add_f32 / (libdevice.exp(-add_f32) + 1.0)
    silu_bf16 = silu.to(tl.bfloat16)
    scaled = (silu_bf16.to(tl.float32) * FINAL_SCALE_).to(tl.bfloat16)
    return add_value, scaled


@triton.jit
def _nfnet_gate_silu_avgpool_kernel(
    gate_ptr,
    x_ptr,
    residual_ptr,
    add_ptr,
    activation_ptr,
    pool_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    RS2: tl.constexpr,
    RS3: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    AS3: tl.constexpr,
    MS0: tl.constexpr,
    MS1: tl.constexpr,
    MS2: tl.constexpr,
    MS3: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    PS2: tl.constexpr,
    PS3: tl.constexpr,
    FINAL_SCALE_: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    out_offsets = tl.program_id(1) * BLOCK_O + tl.arange(0, BLOCK_O)
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)

    out_h: tl.constexpr = H // 2
    out_w: tl.constexpr = W // 2
    out_hw: tl.constexpr = out_h * out_w
    out_mask = out_offsets < out_hw
    c_mask = c_offsets < C
    mask = out_mask[:, None] & c_mask[None, :]

    oh = out_offsets // out_w
    ow = out_offsets - oh * out_w
    ih0 = oh * 2
    iw0 = ow * 2

    gate = tl.load(gate_ptr + n * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    gate_bf16 = (1.0 / (libdevice.exp(-gate) + 1.0)).to(tl.bfloat16)

    pool_acc = tl.zeros((BLOCK_O, BLOCK_C), dtype=tl.float32)
    for dh in tl.static_range(0, 2):
        ih = ih0 + dh
        for dw in tl.static_range(0, 2):
            iw = iw0 + dw
            x_offsets = n * XS0 + c_offsets[None, :] * XS1 + ih[:, None] * XS2 + iw[:, None] * XS3
            residual_offsets = n * RS0 + c_offsets[None, :] * RS1 + ih[:, None] * RS2 + iw[:, None] * RS3
            add_offsets = n * AS0 + c_offsets[None, :] * AS1 + ih[:, None] * AS2 + iw[:, None] * AS3
            activation_offsets = n * MS0 + c_offsets[None, :] * MS1 + ih[:, None] * MS2 + iw[:, None] * MS3

            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
            add_value, activation = _producer_values(gate_bf16[None, :], x, residual, FINAL_SCALE_)
            tl.store(add_ptr + add_offsets, add_value, mask=mask)
            tl.store(activation_ptr + activation_offsets, activation, mask=mask)
            pool_acc += activation.to(tl.float32)

    pool_offsets = n * PS0 + c_offsets[None, :] * PS1 + oh[:, None] * PS2 + ow[:, None] * PS3
    pool = (pool_acc * 0.25).to(tl.bfloat16)
    tl.store(pool_ptr + pool_offsets, pool, mask=mask)


def _launch(inputs, *, BLOCK_O: int, BLOCK_C: int, num_warps: int):
    gate, x, residual = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    out_h = h // 2
    out_w = w // 2

    full_stride = tuple(int(s) for s in x.stride())
    pool_stride = (c * out_h * out_w, 1, out_w * c, c)
    add = torch.empty_strided(
        (n, c, h, w),
        full_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    activation = torch.empty_strided(
        (n, c, h, w),
        full_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    pool = torch.empty_strided(
        (n, c, out_h, out_w),
        pool_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )

    xs = tuple(int(s) for s in x.stride())
    rs = tuple(int(s) for s in residual.stride())
    grid = (n, triton.cdiv(out_h * out_w, BLOCK_O), triton.cdiv(c, BLOCK_C))
    _nfnet_gate_silu_avgpool_kernel[grid](
        gate,
        x,
        residual,
        add,
        activation,
        pool,
        C=c,
        H=h,
        W=w,
        XS0=xs[0],
        XS1=xs[1],
        XS2=xs[2],
        XS3=xs[3],
        RS0=rs[0],
        RS1=rs[1],
        RS2=rs[2],
        RS3=rs[3],
        AS0=full_stride[0],
        AS1=full_stride[1],
        AS2=full_stride[2],
        AS3=full_stride[3],
        MS0=full_stride[0],
        MS1=full_stride[1],
        MS2=full_stride[2],
        MS3=full_stride[3],
        PS0=pool_stride[0],
        PS1=pool_stride[1],
        PS2=pool_stride[2],
        PS3=pool_stride[3],
        FINAL_SCALE_=FINAL_SCALE,
        BLOCK_O=BLOCK_O,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return add, activation, pool


@oracle_impl(hardware="B200", point="b2d12e6c", BLOCK_O=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="a4a4052f", BLOCK_O=8, BLOCK_C=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_O: int, BLOCK_C: int, num_warps: int):
    return _launch(inputs, BLOCK_O=BLOCK_O, BLOCK_C=BLOCK_C, num_warps=num_warps)
