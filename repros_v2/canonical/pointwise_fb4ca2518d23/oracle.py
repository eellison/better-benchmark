"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet gate/residual SiLU scope in one stride-specialized Triton pointwise kernel, including the singleton-spatial sigmoid broadcast, the returned bf16 residual-add tensor, the explicit fp32 SiLU expression, the bf16 cast before the final scale, and the final bf16 output layout for every point, whereas Inductor lowers the same graph as generic fused pointwise work over an expanded channels-last broadcast; Inductor cannot do this today because its pointwise scheduler/codegen has no B200-tuned multi-output singleton-gate broadcast template that reuses the channel gate and emits both dependent bf16 outputs while preserving the captured dtype-rounding boundaries; the fix is SCHEDULER_FUSION: teach pointwise scheduling to specialize dense channel-gate broadcasts with side-output stores and fused activation epilogues."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


FINAL_SCALE = 0.9622504486493761


@triton.jit
def _nfnet_gate_silu_kernel(
    gate_ptr,
    x_ptr,
    residual_ptr,
    add_out_ptr,
    silu_out_ptr,
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
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    OS3: tl.constexpr,
    FINAL_SCALE_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = H * W
    hw_mask = hw_offsets < hw
    c_mask = c_offsets < C
    mask = hw_mask[:, None] & c_mask[None, :]

    h_offsets = hw_offsets // W
    w_offsets = hw_offsets - h_offsets * W

    gate = tl.load(gate_ptr + n * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    gate_bf16 = tl.sigmoid(gate).to(tl.bfloat16)

    x_offsets = (
        n * XS0
        + c_offsets[None, :] * XS1
        + h_offsets[:, None] * XS2
        + w_offsets[:, None] * XS3
    )
    residual_offsets = (
        n * RS0
        + c_offsets[None, :] * RS1
        + h_offsets[:, None] * RS2
        + w_offsets[:, None] * RS3
    )
    out_offsets = (
        n * OS0
        + c_offsets[None, :] * OS1
        + h_offsets[:, None] * OS2
        + w_offsets[:, None] * OS3
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)

    mul0 = (x * gate_bf16[None, :].to(tl.float32)).to(tl.bfloat16)
    mul1 = (mul0.to(tl.float32) * 2.0).to(tl.bfloat16)
    mul2 = (mul1.to(tl.float32) * 0.2).to(tl.bfloat16)
    add_value = (mul2.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(add_out_ptr + out_offsets, add_value, mask=mask)

    add_f32 = add_value.to(tl.float32)
    silu = add_f32 / (libdevice.exp(-add_f32) + 1.0)
    silu_bf16 = silu.to(tl.bfloat16)
    out = silu_bf16.to(tl.float32) * FINAL_SCALE_
    tl.store(silu_out_ptr + out_offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="53dca1d5", BLOCK_HW=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="c972bcba", BLOCK_HW=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="8eccb2bf", BLOCK_HW=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="9983a35a", BLOCK_HW=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="b2d12e6c", BLOCK_HW=8, BLOCK_C=128, num_warps=4)
@oracle_impl(hardware="B200", point="a4a4052f", BLOCK_HW=8, BLOCK_C=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int, num_warps: int):
    gate, x, residual = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    out_stride = tuple(int(s) for s in x.stride())
    add_out = torch.empty_strided(
        (n, c, h, w),
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    silu_out = torch.empty_strided(
        (n, c, h, w),
        out_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )
    xs = tuple(int(s) for s in x.stride())
    rs = tuple(int(s) for s in residual.stride())

    grid = (n, triton.cdiv(h * w, BLOCK_HW), triton.cdiv(c, BLOCK_C))
    _nfnet_gate_silu_kernel[grid](
        gate,
        x,
        residual,
        add_out,
        silu_out,
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
        OS0=out_stride[0],
        OS1=out_stride[1],
        OS2=out_stride[2],
        OS3=out_stride[3],
        FINAL_SCALE_=FINAL_SCALE,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return add_out, silu_out
