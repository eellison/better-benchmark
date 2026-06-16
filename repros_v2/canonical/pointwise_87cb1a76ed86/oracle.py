"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete PyTorch-UNet bf16 BN-inference affine, explicit bf16 cast, NaN-preserving ReLU, bilinear-index resize, right-column pad, and final channel-cat output with Triton kernels that write the final `[1,256,320,479]` tensor directly, whereas Inductor lowers the broadcast normalization producer, indexed interpolation, padding, and cat through generic pointwise/layout materializations; Inductor cannot do this today because the scheduler treats the bilinear `_unsafe_index` stencil and pad/cat layout as fusion barriers and does not sink the channel affine producer through the final output store while preserving the bf16 cast boundary; the fix is SCHEDULER_FUSION: allow pointwise channel producers to fuse through affine bilinear-index materialization kernels and emit the padded cat output directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


CHANNELS = 128
IN_H = 160
IN_W = 239
OUT_H = 320
OUT_W_INTERP = 478
OUT_W = 479
IN_HW = IN_H * IN_W
OUT_HW = OUT_H * OUT_W
COPY_ELEMS = CHANNELS * OUT_HW


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
def _copy_first_half_kernel(src_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _bn_relu_sample(x_ptr, mean, invstd, weight, bias, offsets, mask, zero_bf16):
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    affine_bf16 = affine.to(tl.bfloat16)
    relu_bf16 = tl.where(affine_bf16 < zero_bf16, zero_bf16, affine_bf16)
    return relu_bf16.to(tl.float32)


@triton.jit
def _resize_pad_second_half_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW_INTERP: tl.constexpr,
    OW: tl.constexpr,
    IHW: tl.constexpr,
    OHW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    w_block = tl.program_id(1)

    c = pid // OH
    oh = pid - c * OH
    ow = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = ow < OW
    interp_mask = ow < OW_INTERP
    valid = mask & interp_mask

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    invstd = 1.0 / libdevice.sqrt(_f32_add(var, 1.0e-5))

    src_h = _f32_mul(oh.to(tl.float32), 0.49843260188087773)
    src_h = tl.maximum(src_h, 0.0)
    h_lo = src_h.to(tl.int64)
    h_hi = tl.minimum(h_lo + 1, IH - 1)
    h_frac = _f32_sub(src_h, h_lo.to(tl.float32))
    h_frac = tl.minimum(tl.maximum(h_frac, 0.0), 1.0)

    src_w = _f32_mul(ow.to(tl.float32), 0.4989517819706499)
    src_w = tl.maximum(src_w, 0.0)
    w_lo = src_w.to(tl.int64)
    w_hi = tl.minimum(w_lo + 1, IW - 1)
    w_frac = _f32_sub(src_w, w_lo.to(tl.float32))
    w_frac = tl.minimum(tl.maximum(w_frac, 0.0), 1.0)

    base = c * IHW
    zero_bf16 = tl.full((BLOCK_W,), 0.0, tl.bfloat16)
    v_hi_hi = _bn_relu_sample(
        x_ptr, mean, invstd, weight, bias, base + h_hi * IW + w_hi, valid, zero_bf16
    )
    v_hi_lo = _bn_relu_sample(
        x_ptr, mean, invstd, weight, bias, base + h_hi * IW + w_lo, valid, zero_bf16
    )
    v_lo_hi = _bn_relu_sample(
        x_ptr, mean, invstd, weight, bias, base + h_lo * IW + w_hi, valid, zero_bf16
    )
    v_lo_lo = _bn_relu_sample(
        x_ptr, mean, invstd, weight, bias, base + h_lo * IW + w_lo, valid, zero_bf16
    )

    hi_row = _f32_add(v_hi_lo, _f32_mul(_f32_sub(v_hi_hi, v_hi_lo), w_frac))
    lo_row = _f32_add(v_lo_lo, _f32_mul(_f32_sub(v_lo_hi, v_lo_lo), w_frac))
    value = _f32_add(lo_row, _f32_mul(_f32_sub(hi_row, lo_row), h_frac))
    value = tl.where(interp_mask, value, 0.0)

    out_offsets = (C + c) * OHW + oh * OW + ow
    tl.store(out_ptr + out_offsets, value.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="f4799916", COPY_BLOCK=1024, BLOCK_W=256, copy_warps=4, resize_warps=4)
def oracle_forward(inputs, *, COPY_BLOCK, BLOCK_W, copy_warps, resize_warps):
    mean, x, var, weight, bias, skip, _shape = inputs
    out = torch.empty((1, 2 * CHANNELS, OUT_H, OUT_W), device=x.device, dtype=torch.bfloat16)
    _copy_first_half_kernel[(triton.cdiv(COPY_ELEMS, COPY_BLOCK),)](
        skip,
        out,
        n_elements=COPY_ELEMS,
        BLOCK=COPY_BLOCK,
        num_warps=copy_warps,
        num_stages=3,
    )
    _resize_pad_second_half_kernel[(CHANNELS * OUT_H, triton.cdiv(OUT_W, BLOCK_W))](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        C=CHANNELS,
        IH=IN_H,
        IW=IN_W,
        OH=OUT_H,
        OW_INTERP=OUT_W_INTERP,
        OW=OUT_W,
        IHW=IN_HW,
        OHW=OUT_HW,
        BLOCK_W=BLOCK_W,
        num_warps=resize_warps,
        num_stages=3,
    )
    return out
