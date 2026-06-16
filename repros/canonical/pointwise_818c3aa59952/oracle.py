"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 BN-inference affine returned by Repro.forward for all captured channels-last and contiguous NCHW points, hoisting only the per-channel `reciprocal(sqrt(var + eps))` producer and then applying the elementwise chain in the original `(x - mean) * invstd * weight + bias` f32 order before the final bf16 cast; Inductor currently emits a generic pointwise kernel that recomputes the same channel-invariant sqrt/reciprocal for every output element; Inductor cannot do this today because its pointwise scheduler does not canonicalize broadcast-invariant BN parameters into per-channel producers while preserving the captured cast and operation order; the fix is ALGEBRAIC_ELIMINATION: hoist invariant BN-inference parameter work and reuse it in the output writer without changing output scope, dtype, or strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _invstd_kernel(var_ptr, invstd_ptr, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < C
    var = tl.load(var_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    sqrt = tl.sqrt(var + EPS)
    invstd = 1.0 / sqrt
    tl.store(invstd_ptr + offsets, invstd, mask=mask)


@triton.jit
def _bn_affine_kernel(
    mean_ptr,
    x_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    if CHANNELS_LAST:
        c = offsets % C
    else:
        c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    y = x - mean
    y = y * invstd
    y = y * weight
    y = y + bias
    tl.store(out_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="c6312c18", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="1a06d794", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="e9bc5ee0", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="701aa9ad", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="5540c28a", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="4c825568", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="4280fd00", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="654b8cd3", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="a118ca10", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="a047c97b", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="76c62ba9", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="0a96753f", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="4c724370", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="ef49f65d", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="c68fd75d", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="ecb9a0d7", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="3ddafcba", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="3350df9a", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="e987d149", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="1fca60d1", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="8e74e392", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="b72bb793", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="def42dcc", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="d6d99242", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="9f949812", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="6129a191", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="5959a6c3", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="bd03f6f7", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="6349ba76", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="e2542415", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="d132f392", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="4552b8d5", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="40a7d00c", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="5e9ad3e0", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="4f4eacf9", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="7395f576", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="908bdc99", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="d5e26824", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="05028dd8", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="261d5bdd", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="aaa0fce0", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="ced04216", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="d773a707", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="d820ae29", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="fa15eadf", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="6e30cacc", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="017f27d2", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="5c1756a4", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="13e72620", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="f3036c42", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="9a05b224", BLOCK=1024, COEFF_BLOCK=512)
@oracle_impl(hardware="B200", point="0ba75884", BLOCK=1024, COEFF_BLOCK=512)
def oracle_forward(inputs, *, BLOCK, COEFF_BLOCK):
    mean, x, var, weight, bias = inputs
    n, c, h, w = x.shape
    total = n * c * h * w
    hw = h * w
    channels_last = x.stride(1) == 1

    invstd = torch.empty((c,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    _invstd_kernel[(1,)](
        var,
        invstd,
        C=c,
        EPS=1.0e-5,
        BLOCK=COEFF_BLOCK,
        num_warps=8,
    )
    _bn_affine_kernel[(triton.cdiv(total, BLOCK),)](
        mean,
        x,
        invstd,
        weight,
        bias,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        CHANNELS_LAST=channels_last,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out
