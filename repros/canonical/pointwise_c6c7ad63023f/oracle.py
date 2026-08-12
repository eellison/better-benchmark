"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet virtual channel concat, f32 batchnorm-inference affine with `sqrt(var + 1e-5)`, explicit bf16 cast, and NaN-preserving ReLU directly into the returned contiguous bf16 tensor for all four captured channel/spatial points, whereas Inductor materializes the fixed channel cat separately from the downstream broadcast affine and ReLU pointwise work; Inductor cannot do this today because scheduler/codegen does not model static channel concat as a virtual multi-source producer that can be sunk through per-channel affine consumers while preserving the bf16 cast-before-ReLU boundary; the fix is SCHEDULER_FUSION: allow fixed-shape channel concat producers to feed fused pointwise consumers with guarded source selection and exact dtype-boundary preservation."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _relu_after_bf16(x):
    xb = x.to(tl.bfloat16).to(tl.float32)
    return tl.where((xb > 0.0) | (xb != xb), xb, 0.0)


@triton.jit
def _virtual_cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C0: tl.constexpr,
    COUT: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    spatial = offsets % HW
    channel = (offsets // HW) % COUT
    batch = offsets // (COUT * HW)

    rel = channel - C0
    in0 = channel < C0
    in1 = (rel >= 0) & (rel < 32)
    in2 = (rel >= 32) & (rel < 64)
    in3 = rel >= 64

    c0 = tl.where(in0, channel, 0)
    c1 = tl.where(in1, rel, 0)
    c2 = tl.where(in2, rel - 32, 0)
    c3 = tl.where(in3, rel - 64, 0)

    x0 = tl.load(x0_ptr + batch * C0 * HW + c0 * HW + spatial, mask=mask & in0, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + batch * 32 * HW + c1 * HW + spatial, mask=mask & in1, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + batch * 32 * HW + c2 * HW + spatial, mask=mask & in2, other=0.0).to(tl.float32)
    x3 = tl.load(x3_ptr + batch * 32 * HW + c3 * HW + spatial, mask=mask & in3, other=0.0).to(tl.float32)
    x = x0 + x1 + x2 + x3

    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * (1.0 / tl.sqrt(var + 1.0e-5)) * weight + bias
    tl.store(out_ptr + offsets, _relu_after_bf16(y), mask=mask)


@oracle_impl(hardware="B200", point="7ea3cc48", C0=512, COUT=608, HW=49, BLOCK=256)
@oracle_impl(hardware="B200", point="6f284162", C0=256, COUT=352, HW=196, BLOCK=256)
@oracle_impl(hardware="B200", point="9cd6bc43", C0=128, COUT=224, HW=784, BLOCK=256)
@oracle_impl(hardware="B200", point="ea04eb28", C0=64, COUT=160, HW=3136, BLOCK=256)
def oracle_forward(inputs, *, C0: int, COUT: int, HW: int, BLOCK: int):
    x0, x1, x2, x3, mean, var, weight, bias = inputs
    total = 64 * COUT * HW
    out = torch.empty_strided((64, COUT, HW), (COUT * HW, HW, 1), device=x0.device, dtype=torch.bfloat16)
    _virtual_cat_bn_relu_kernel[(triton.cdiv(total, BLOCK),)](
        x0,
        x1,
        x2,
        x3,
        mean,
        var,
        weight,
        bias,
        out,
        TOTAL=total,
        C0=C0,
        COUT=COUT,
        HW=HW,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=3,
    )
    if HW == 49:
        return out.view(64, COUT, 7, 7)
    if HW == 196:
        return out.view(64, COUT, 14, 14)
    if HW == 784:
        return out.view(64, COUT, 28, 28)
    return out.view(64, COUT, 56, 56)
