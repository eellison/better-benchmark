"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet singleton sigmoid gate, scalar scale, residual add, exact-erf GELU scale, returned residual-add side output, returned full-resolution scaled-GELU output, and fixed 2x2 stride-2 avg_pool2d output in one Triton output-stencil kernel that reuses each rounded bf16 activation value for the pooled sibling output, whereas Inductor materializes the returned pointwise outputs with generic scheduling and then runs avg_pool2d as a separate consumer; Inductor cannot do this today because scheduler fusion does not sink a visible multi-output broadcast pointwise producer through a fixed-window pooling consumer while preserving bf16/fp32 cast boundaries and channels-last strides; the fix is SCHEDULER_FUSION: teach fixed-window pooling codegen to fuse broadcast pointwise producer DAGs into multi-output stencil schedules that emit all visible side outputs and the pooled output from the same plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


FINAL_SCALE = 0.8980265101338745


@triton.jit
def _producer_values(gate_bf16, x, residual, scalar_bf16, FINAL_SCALE_: tl.constexpr):
    mul0 = (x * gate_bf16.to(tl.float32)).to(tl.bfloat16)
    mul1 = (mul0.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul2 = (mul1.to(tl.float32) * scalar_bf16.to(tl.float32)).to(tl.bfloat16)
    mul3 = (mul2.to(tl.float32) * 0.2).to(tl.bfloat16)
    add_value = (mul3.to(tl.float32) + residual).to(tl.bfloat16)

    add_f32 = add_value.to(tl.float32)
    gelu = (add_f32 * 0.5) * (libdevice.erf(add_f32 * 0.7071067811865476) + 1.0)
    gelu_bf16 = gelu.to(tl.bfloat16)
    scaled = (gelu_bf16.to(tl.float32) * 1.7015043497085571).to(tl.bfloat16)
    final_value = (scaled.to(tl.float32) * FINAL_SCALE_).to(tl.bfloat16)
    return add_value, final_value


@triton.jit
def _nfnet_gate_gelu_pool_kernel(
    gate_ptr,
    payload_ptr,
    scalar_ptr,
    residual_ptr,
    add_out_ptr,
    final_out_ptr,
    pool_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    PS2: tl.constexpr,
    PS3: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    RS2: tl.constexpr,
    RS3: tl.constexpr,
    FS0: tl.constexpr,
    FS1: tl.constexpr,
    FS2: tl.constexpr,
    FS3: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    OS3: tl.constexpr,
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
            full_offsets = (
                n * PS0
                + c_offsets[None, :] * PS1
                + ih[:, None] * PS2
                + iw[:, None] * PS3
            )
            residual_offsets = (
                n * RS0
                + c_offsets[None, :] * RS1
                + ih[:, None] * RS2
                + iw[:, None] * RS3
            )
            final_offsets = (
                n * FS0
                + c_offsets[None, :] * FS1
                + ih[:, None] * FS2
                + iw[:, None] * FS3
            )
            payload = tl.load(payload_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
            add_value, final_value = _producer_values(
                gate_bf16[None, :],
                payload,
                residual,
                scalar_bf16,
                FINAL_SCALE_,
            )
            tl.store(add_out_ptr + final_offsets, add_value, mask=mask)
            tl.store(final_out_ptr + final_offsets, final_value, mask=mask)
            pool_acc += final_value.to(tl.float32)

    pool_offsets = (
        n * OS0
        + c_offsets[None, :] * OS1
        + oh[:, None] * OS2
        + ow[:, None] * OS3
    )
    pool_value = (pool_acc * 0.25).to(tl.bfloat16)
    tl.store(pool_ptr + pool_offsets, pool_value, mask=mask)


def _launch(inputs, *, BLOCK_O: int, BLOCK_C: int, num_warps: int):
    gate, payload, scalar, residual = inputs
    n, c, h, w = (int(dim) for dim in payload.shape)
    out_h = h // 2
    out_w = w // 2

    full_stride = tuple(int(s) for s in payload.stride())
    pool_stride = (c * out_h * out_w, 1, out_w * c, c)
    add_out = torch.empty_strided(
        (n, c, h, w),
        full_stride,
        device=payload.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        (n, c, h, w),
        full_stride,
        device=payload.device,
        dtype=torch.bfloat16,
    )
    pool_out = torch.empty_strided(
        (n, c, out_h, out_w),
        pool_stride,
        device=payload.device,
        dtype=torch.bfloat16,
    )

    residual_stride = tuple(int(s) for s in residual.stride())
    grid = (n, triton.cdiv(out_h * out_w, BLOCK_O), triton.cdiv(c, BLOCK_C))
    _nfnet_gate_gelu_pool_kernel[grid](
        gate,
        payload,
        scalar,
        residual,
        add_out,
        final_out,
        pool_out,
        C=c,
        H=h,
        W=w,
        PS0=full_stride[0],
        PS1=full_stride[1],
        PS2=full_stride[2],
        PS3=full_stride[3],
        RS0=residual_stride[0],
        RS1=residual_stride[1],
        RS2=residual_stride[2],
        RS3=residual_stride[3],
        FS0=full_stride[0],
        FS1=full_stride[1],
        FS2=full_stride[2],
        FS3=full_stride[3],
        OS0=pool_stride[0],
        OS1=pool_stride[1],
        OS2=pool_stride[2],
        OS3=pool_stride[3],
        FINAL_SCALE_=FINAL_SCALE,
        BLOCK_O=BLOCK_O,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return add_out, final_out, pool_out


# 6498d204: (T([128,1536,1,1], bf16), T([128,1536,12,12], bf16, stride=(221184,1,18432,1536)), T([], f32), T([128,1536,12,12], bf16, stride=(221184,1,18432,1536)))
@oracle_impl(hardware="B200", point="6498d204", BLOCK_O=8, BLOCK_C=128, num_warps=4)
# b99b11a3: (T([128,512,1,1], bf16), T([128,512,24,24], bf16, stride=(294912,1,12288,512)), T([], f32), T([128,512,24,24], bf16, stride=(294912,1,12288,512)))
@oracle_impl(hardware="B200", point="b99b11a3", BLOCK_O=8, BLOCK_C=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_O: int, BLOCK_C: int, num_warps: int):
    return _launch(inputs, BLOCK_O=BLOCK_O, BLOCK_C=BLOCK_C, num_warps=num_warps)
