"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GhostNet dual BN-inference affine, virtual channel-cat, and residual add scope by hoisting only the captured per-channel reciprocal-sqrt terms while preserving the original f32 subtraction/multiply/add order, both bf16 branch cast boundaries, the final bf16 add, and the channels-last output layout, whereas Inductor lowers the same graph as a generic pointwise/cat kernel with broadcast scalar work embedded in the output loop; Inductor cannot do this today because its pointwise simplifier does not canonicalize fixed BN-inference affine producers through a channel cat while retaining exact bf16 conversion boundaries and NaN behavior; the fix is ALGEBRAIC_ELIMINATION: expose the per-channel reciprocal-sqrt values and virtualize the cat before codegen so the final writer can use the original operation order with less redundant scalar work."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _invstd_kernel(
    var0_ptr,
    var1_ptr,
    inv0_ptr,
    inv1_ptr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.arange(0, BLOCK)
    mask1 = c < C1
    var1 = tl.load(var1_ptr + c, mask=mask1, other=0.0).to(tl.float32)
    inv1 = _f32_div(1.0, tl.sqrt(_f32_add(var1, 1.0e-5)))
    tl.store(inv1_ptr + c, inv1, mask=mask1)

    mask0 = c < C0
    var0 = tl.load(var0_ptr + c, mask=mask0, other=0.0).to(tl.float32)
    inv0 = _f32_div(1.0, tl.sqrt(_f32_add(var0, 1.0e-5)))
    tl.store(inv0_ptr + c, inv0, mask=mask0)


@triton.jit
def _dual_bn_cat_add_kernel(
    mean0_ptr,
    conv0_ptr,
    inv0_ptr,
    weight0_ptr,
    bias0_ptr,
    residual_ptr,
    mean1_ptr,
    conv1_ptr,
    inv1_ptr,
    weight1_ptr,
    bias1_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c1 = offsets % C1
    tmp = offsets // C1
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    n = tmp // H
    c0 = c1 - C0
    low = c1 < C0

    conv1 = tl.load(conv1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c1, mask=mask, other=0.0).to(tl.float32)
    inv1 = tl.load(inv1_ptr + c1, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c1, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c1, mask=mask, other=0.0).to(tl.float32)
    branch1 = _f32_add(_f32_mul(_f32_mul(_f32_sub(conv1, mean1), inv1), weight1), bias1)
    branch1_bf16 = _round_to_bf16_f32(branch1)

    residual_offsets = ((n * H + h) * W + w) * C0 + c1
    residual = tl.load(residual_ptr + residual_offsets, mask=mask & low, other=0.0).to(tl.float32)

    conv0_offsets = ((n * H + h) * W + w) * C0 + c0
    conv0 = tl.load(conv0_ptr + conv0_offsets, mask=mask & ~low, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + c0, mask=mask & ~low, other=0.0).to(tl.float32)
    inv0 = tl.load(inv0_ptr + c0, mask=mask & ~low, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + c0, mask=mask & ~low, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + c0, mask=mask & ~low, other=0.0).to(tl.float32)
    branch0 = _f32_add(_f32_mul(_f32_mul(_f32_sub(conv0, mean0), inv0), weight0), bias0)
    branch0_bf16 = _round_to_bf16_f32(branch0)

    cat_value = tl.where(low, residual, branch0_bf16)
    out = _round_to_bf16_f32(_f32_add(cat_value, branch1_bf16))
    tl.store(out_ptr + offsets, out, mask=mask)


def _forward(inputs, *, C0, H, BLOCK):
    mean0, conv0, var0, weight0, bias0, residual, mean1, conv1, var1, weight1, bias1 = inputs
    n = int(conv1.shape[0])
    c1 = C0 * 2
    total = n * c1 * H * H
    out = torch.empty_strided(tuple(conv1.shape), tuple(conv1.stride()), device=conv1.device, dtype=torch.bfloat16)
    inv0 = torch.empty_strided((C0,), (1,), device=conv1.device, dtype=torch.float32)
    inv1 = torch.empty_strided((c1,), (1,), device=conv1.device, dtype=torch.float32)
    _invstd_kernel[(1,)](
        var0,
        var1,
        inv0,
        inv1,
        C0=C0,
        C1=c1,
        BLOCK=triton.next_power_of_2(c1),
        num_warps=1,
        num_stages=1,
    )
    _dual_bn_cat_add_kernel[(triton.cdiv(total, BLOCK),)](
        mean0,
        conv0,
        inv0,
        weight0,
        bias0,
        residual,
        mean1,
        conv1,
        inv1,
        weight1,
        bias1,
        out,
        TOTAL=total,
        C0=C0,
        C1=c1,
        H=H,
        W=H,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="0bcbb75d", C0=80, H=7, BLOCK=1024)
@oracle_impl(hardware="B200", point="517be145", C0=56, H=14, BLOCK=1024)
@oracle_impl(hardware="B200", point="6eba313b", C0=40, H=14, BLOCK=1024)
@oracle_impl(hardware="B200", point="4db2a61e", C0=20, H=28, BLOCK=1024)
@oracle_impl(hardware="B200", point="aec27f0f", C0=12, H=56, BLOCK=1024)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
