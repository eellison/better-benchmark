"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet sigmoid gate, scalar scale, residual add, exact-erf GELU scale, returned full-resolution activation, and fixed 2x2 stride-2 avg_pool2d scope in one Triton output-stencil kernel that reuses each rounded bf16 activation value for the pooled sibling output, whereas Inductor materializes the returned activation with generic pointwise scheduling and then runs a separate avg_pool2d consumer over that buffer; Inductor cannot do this today because scheduler fusion does not sink a returned broadcast-gated pointwise producer through a fixed-window pooling consumer while preserving visible side outputs and bf16/fp32 cast boundaries; the fix is SCHEDULER_FUSION: teach fixed-window pooling codegen to fuse broadcast pointwise producer DAGs into a multi-output stencil plan that emits both the full activation stores and pooled stores from the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


FINAL_SCALE = 0.8980265101338745


@triton.jit
def _producer_value(gate_bf16, x, residual, scalar_bf16, FINAL_SCALE_: tl.constexpr):
    mul0 = (x * gate_bf16.to(tl.float32)).to(tl.bfloat16)
    mul1 = (mul0.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul2 = (mul1.to(tl.float32) * scalar_bf16.to(tl.float32)).to(tl.bfloat16)
    mul3 = (mul2.to(tl.float32) * 0.2).to(tl.bfloat16)
    add_value = (mul3.to(tl.float32) + residual).to(tl.bfloat16)

    add_f32 = add_value.to(tl.float32)
    gelu = (add_f32 * 0.5) * (libdevice.erf(add_f32 * 0.7071067811865476) + 1.0)
    gelu_bf16 = gelu.to(tl.bfloat16)
    scaled = (gelu_bf16.to(tl.float32) * 1.7015043497085571).to(tl.bfloat16)
    return (scaled.to(tl.float32) * FINAL_SCALE_).to(tl.bfloat16)


@triton.jit
def _nfnet_gelu_avgpool_kernel(
    gate_ptr,
    x_ptr,
    scalar_ptr,
    residual_ptr,
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
    gate_bf16 = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16)
    scalar_bf16 = tl.load(scalar_ptr).to(tl.bfloat16)

    pool_acc = tl.zeros((BLOCK_O, BLOCK_C), dtype=tl.float32)
    for dh in tl.static_range(0, 2):
        ih = ih0 + dh
        for dw in tl.static_range(0, 2):
            iw = iw0 + dw
            x_offsets = (
                n * XS0
                + c_offsets[None, :] * XS1
                + ih[:, None] * XS2
                + iw[:, None] * XS3
            )
            residual_offsets = (
                n * RS0
                + c_offsets[None, :] * RS1
                + ih[:, None] * RS2
                + iw[:, None] * RS3
            )
            activation_offsets = (
                n * AS0
                + c_offsets[None, :] * AS1
                + ih[:, None] * AS2
                + iw[:, None] * AS3
            )
            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
            activation = _producer_value(
                gate_bf16[None, :],
                x,
                residual,
                scalar_bf16,
                FINAL_SCALE_,
            )
            tl.store(activation_ptr + activation_offsets, activation, mask=mask)
            pool_acc += activation.to(tl.float32)

    pool_offsets = (
        n * PS0
        + c_offsets[None, :] * PS1
        + oh[:, None] * PS2
        + ow[:, None] * PS3
    )
    pool = (pool_acc * 0.25).to(tl.bfloat16)
    tl.store(pool_ptr + pool_offsets, pool, mask=mask)


def _launch(inputs, *, BLOCK_O: int, BLOCK_C: int, num_warps: int):
    gate, x, scalar, residual = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    out_h = h // 2
    out_w = w // 2

    activation_stride = tuple(int(s) for s in x.stride())
    pool_stride = (c * out_h * out_w, 1, out_w * c, c)
    activation = torch.empty_strided(
        (n, c, h, w),
        activation_stride,
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
    _nfnet_gelu_avgpool_kernel[grid](
        gate,
        x,
        scalar,
        residual,
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
        AS0=activation_stride[0],
        AS1=activation_stride[1],
        AS2=activation_stride[2],
        AS3=activation_stride[3],
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
    return activation, pool


# 27a04b02: (T([128,1536,1,1], bf16), T([128,1536,16,16], bf16, stride=(393216,1,24576,1536)), T([], bf16), T([128,1536,16,16], bf16, stride=(393216,1,24576,1536)))
@oracle_impl(hardware="B200", point="27a04b02", BLOCK_O=8, BLOCK_C=128, num_warps=4)
# a4fc89e1: (T([128,512,1,1], bf16), T([128,512,32,32], bf16, stride=(524288,1,16384,512)), T([], bf16), T([128,512,32,32], bf16, stride=(524288,1,16384,512)))
@oracle_impl(hardware="B200", point="a4fc89e1", BLOCK_O=8, BLOCK_C=128, num_warps=4)
# 945b3208: (T([128,256,1,1], bf16), T([128,256,64,64], bf16, stride=(1048576,1,16384,256)), T([], bf16), T([128,256,64,64], bf16, stride=(1048576,1,16384,256)))
@oracle_impl(hardware="B200", point="945b3208", BLOCK_O=8, BLOCK_C=128, num_warps=4)
# 1d754e93: (T([128,256,1,1], bf16), T([128,256,48,48], bf16, stride=(589824,1,12288,256)), T([], f32), T([128,256,48,48], bf16, stride=(589824,1,12288,256)))
@oracle_impl(hardware="B200", point="1d754e93", BLOCK_O=8, BLOCK_C=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_O: int, BLOCK_C: int, num_warps: int):
    return _launch(inputs, BLOCK_O=BLOCK_O, BLOCK_C=BLOCK_C, num_warps=num_warps)
