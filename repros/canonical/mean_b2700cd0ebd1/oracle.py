"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet SiLU spatial-mean scope in one channels-last Triton reduction, including bf16 input promotion, natural-exp fp32 SiLU as `x / (exp(-x) + 1)`, the required bf16 rounding boundary before the mean, fp32 accumulation of the rounded activation over the fixed spatial tile, final bf16 mean rounding, and the returned contiguous `[128,2304]` view, whereas Inductor lowers the decomposed cast/SiLU/cast/mean/as_strided/view graph through its generic fused reduction scheduler; Inductor cannot do this today because scheduler/codegen has no guarded B200 template for a bf16-rounded pointwise producer feeding a small fixed spatial mean over channels-last NFNet tensors while directly emitting the final view layout; the fix is SCHEDULER_FUSION: teach reduction scheduling to sink explicit bf16 activation producers into small-spatial mean templates and preserve the cast boundaries while storing the viewed output directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

from oracle_harness import oracle_impl


@triton.jit
def _rounded_silu(x):
    denom = tl_math.exp(-x) + 1.0
    return (x / denom).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _silu_bf16_spatial_mean_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS3: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = tl.arange(0, BLOCK_HW)

    c_mask = c_offsets < C
    hw_mask = hw_offsets < H * W
    mask = hw_mask[:, None] & c_mask[None, :]

    x_offsets = (
        n * XS0
        + c_offsets[None, :] * XS1
        + hw_offsets[:, None] * XS3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    rounded = _rounded_silu(x)
    total = tl.sum(tl.where(hw_mask[:, None], rounded, 0.0), axis=0)
    mean = total / tl.full((BLOCK_C,), H * W, tl.float32)

    out_offsets = n * C + c_offsets
    tl.store(out_ptr + out_offsets, mean.to(tl.bfloat16), mask=c_mask & (n < N))


def _launch(inputs, *, BLOCK_C: int, BLOCK_HW: int, num_warps: int):
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    n, c, h, w = (int(dim) for dim in x.shape)
    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    xs = tuple(int(stride) for stride in x.stride())
    grid = (n, triton.cdiv(c, BLOCK_C))
    _silu_bf16_spatial_mean_kernel[grid](
        x,
        out,
        N=n,
        C=c,
        H=h,
        W=w,
        XS0=xs[0],
        XS1=xs[1],
        XS3=xs[3],
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


# ec934f37: (T([128,2304,9,9], bf16, stride=(186624,1,20736,2304)), ...)
@oracle_impl(hardware="B200", point="ec934f37", BLOCK_C=64, BLOCK_HW=128, num_warps=4)
# 9cb825ed: (T([128,2304,7,7], bf16, stride=(112896,1,16128,2304)), ...)
@oracle_impl(hardware="B200", point="9cb825ed", BLOCK_C=32, BLOCK_HW=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_HW: int, num_warps: int):
    return _launch(inputs, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW, num_warps=num_warps)
