"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full DenseNet bf16 BN-affine/ReLU, padded 3x3 stride-2 maxpool value, and downstream BN-affine/ReLU return scope into one Triton stencil that emits both returned bf16 tensors directly, whereas Inductor schedules the pointwise producers/consumers and low-memory maxpool as generic materialized regions; Inductor cannot do this today because scheduler fusion cannot sink affine producers and consumers through prims._low_memory_max_pool_with_offsets while preserving bf16 cast, padding, tie, and NaN semantics plus the visible pooled output; the fix is SCHEDULER_FUSION: add guarded multi-output maxpool stencil fusion that emits pooled values and fused epilogue directly from one loop nest."""

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
def _f32_sqrt(a):
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
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
def _bn_relu_from_inv(x, mean, invstd, weight, bias):
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    rounded = _round_to_bf16_f32(affine)
    return tl.where((rounded > 0.0) | (rounded != rounded), rounded, 0.0)


@triton.jit
def _densenet_bn_pool_bn_kernel(
    mean1_ptr,
    x_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    mean2_ptr,
    var2_ptr,
    weight2_ptr,
    bias2_ptr,
    pool_ptr,
    final_ptr,
    BLOCK_C: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    out_block = tl.program_id(2)

    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    out = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_h = out // 56
    out_w = out - out_h * 56

    c_mask = c < 64
    out_mask = out < 3136
    mask = c_mask[:, None] & out_mask[None, :]

    mean1 = tl.load(mean1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    inv1 = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var1, 1.0e-5))), 1.0)

    best = tl.full((BLOCK_C, BLOCK_OUT), -float("inf"), tl.float32)
    base = (n * 64 + c[:, None]) * 12544

    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh - 1
        valid_h = (in_h >= 0) & (in_h < 112)
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw - 1
            valid = mask & valid_h[None, :] & (in_w[None, :] >= 0) & (in_w[None, :] < 112)
            x = tl.load(
                x_ptr + base + in_h[None, :] * 112 + in_w[None, :],
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            value = _bn_relu_from_inv(
                x,
                mean1[:, None],
                inv1[:, None],
                weight1[:, None],
                bias1[:, None],
            )
            take = valid & ((value > best) | (value != value))
            best = tl.where(take, value, best)

    mean2 = tl.load(mean2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    var2 = tl.load(var2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    inv2 = _f32_mul(_f32_div(1.0, _f32_sqrt(_f32_add(var2, 1.0e-5))), 1.0)
    final = _bn_relu_from_inv(
        best,
        mean2[:, None],
        inv2[:, None],
        weight2[:, None],
        bias2[:, None],
    )

    flat = (n * 64 + c[:, None]) * 3136 + out[None, :]
    tl.store(pool_ptr + flat, best, mask=mask)
    tl.store(final_ptr + flat, final, mask=mask)


@oracle_impl(hardware="B200", point="710a4598", BLOCK_C=4, BLOCK_OUT=128, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_OUT: int, num_warps: int, num_stages: int):
    mean1, x, var1, weight1, bias1, mean2, var2, weight2, bias2, _kernel, _stride = inputs
    pool = torch.empty_strided(
        (64, 64, 56, 56),
        (200704, 3136, 56, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    final = torch.empty_strided(
        (64, 64, 56, 56),
        (200704, 3136, 56, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (64, triton.cdiv(64, BLOCK_C), triton.cdiv(3136, BLOCK_OUT))
    _densenet_bn_pool_bn_kernel[grid](
        mean1,
        x,
        var1,
        weight1,
        bias1,
        mean2,
        var2,
        weight2,
        bias2,
        pool,
        final,
        BLOCK_C=BLOCK_C,
        BLOCK_OUT=BLOCK_OUT,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return pool, final
