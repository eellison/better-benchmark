"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 dual BN-affine/ReLU channel-shuffle scope by preserving the fp32 sqrt/reciprocal/affine math, explicit bf16 cast before ReLU, NaN-preserving ReLU behavior, final cat/view/permute/clone/view layout, and returned split views from one Triton kernel that writes the final interleaved backing allocation directly; Inductor currently lowers the two pointwise BN/ReLU producers and the fixed cat/view/permute/clone/split layout chain as generic scheduled work with avoidable intermediate layout traffic; the fix is SCHEDULER_FUSION: teach scheduler/codegen to keep fixed channel-cat producers virtual through channel-shuffle clones and sink their pointwise producers into the final split backing layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
    ],
    key=["TOTAL"],
)
@triton.jit
def _shuffle_bn_relu_kernel(
    mean0_ptr,
    x0_ptr,
    var0_ptr,
    weight0_ptr,
    bias0_ptr,
    mean1_ptr,
    x1_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    safe_offsets = tl.where(mask, offsets, 0)

    hw = safe_offsets % HW
    c = (safe_offsets // HW) % C
    n = safe_offsets // (C * HW)
    in_offsets = n * C * HW + c * HW + hw

    x0 = tl.load(x0_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var0 = tl.load(var0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    y0 = (x0 - mean0) * (1.0 / tl.sqrt(var0 + EPS))
    y0 = y0 * weight0 + bias0
    y0 = y0.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    y0 = _relu_preserve_nan(y0)

    x1 = tl.load(x1_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    y1 = (x1 - mean1) * (1.0 / tl.sqrt(var1 + EPS))
    y1 = y1 * weight1 + bias1
    y1 = y1.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    y1 = _relu_preserve_nan(y1)

    out_offsets = n * OUT_C * HW + (2 * c) * HW + hw
    tl.store(out_ptr + out_offsets, y0, mask=mask)
    tl.store(out_ptr + out_offsets + HW, y1, mask=mask)


@oracle_impl(hardware="B200", point="5e3665ee")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs

    n = int(arg1_1.shape[0])
    c = int(arg1_1.shape[1])
    h = int(arg1_1.shape[2])
    w = int(arg1_1.shape[3])
    hw = h * w
    out_c = 2 * c
    out = torch.empty((n, out_c, h, w), device=arg1_1.device, dtype=arg1_1.dtype)
    total = n * c * hw

    grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)
    _shuffle_bn_relu_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        out,
        TOTAL=total,
        C=c,
        H=h,
        W=w,
        HW=hw,
        OUT_C=out_c,
        EPS=1.0e-5,
    )

    return out[:, :c, :, :], out[:, c:, :, :]
