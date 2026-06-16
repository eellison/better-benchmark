"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNetV2 bf16 BN-affine-ReLU plus channel-shuffle split scope by writing the two normalized branches directly into the final contiguous shuffled backing tensor and returning the two split views, whereas Inductor materializes the branch results and then schedules the cat/view/permute/clone/view/split layout chain as separate generic work; Inductor cannot do this today because its scheduler does not push fixed channel-shuffle clone indexing back through a virtual channel cat with pointwise producers while preserving the bf16 cast-before-ReLU and split-view output strides; the fix is SCHEDULER_FUSION: teach layout scheduling to sink fixed shuffle/split indexing into fused pointwise stores."""

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
def _relu_bf16(x):
    x = _round_to_bf16_f32(x)
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _bn_relu_shuffle_kernel(
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
    HW: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    hw = offsets % HW
    c = (offsets // HW) % C
    n = offsets // (C * HW)
    src = n * C * HW + c * HW + hw

    x0 = tl.load(x0_ptr + src, mask=mask, other=0.0).to(tl.float32)
    mean0 = tl.load(mean0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var0 = tl.load(var0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight0 = tl.load(weight0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias0 = tl.load(bias0_ptr + c, mask=mask, other=0.0).to(tl.float32)
    inv0 = _f32_mul(_f32_div(1.0, tl.sqrt(_f32_add(var0, 1.0e-5))), 1.0)
    y0 = _f32_mul(_f32_sub(x0, mean0), inv0)
    y0 = _f32_add(_f32_mul(y0, weight0), bias0)
    y0 = _relu_bf16(y0)

    x1 = tl.load(x1_ptr + src, mask=mask, other=0.0).to(tl.float32)
    mean1 = tl.load(mean1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var1 = tl.load(var1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    inv1 = _f32_mul(_f32_div(1.0, tl.sqrt(_f32_add(var1, 1.0e-5))), 1.0)
    y1 = _f32_mul(_f32_sub(x1, mean1), inv1)
    y1 = _f32_add(_f32_mul(y1, weight1), bias1)
    y1 = _relu_bf16(y1)

    out_base = n * OUT_C * HW + (2 * c) * HW + hw
    tl.store(out_ptr + out_base, y0, mask=mask)
    tl.store(out_ptr + out_base + HW, y1, mask=mask)


@oracle_impl(hardware="B200", point="5e3665ee", BLOCK=256)
def oracle_forward(inputs, *, BLOCK):
    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1, shape0, shape1 = inputs
    del shape0
    n = int(x0.shape[0])
    c = int(x0.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    hw = h * w
    out_c = 2 * c
    out = torch.empty_strided(
        tuple(shape1),
        (out_c * hw, hw, w, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    total = n * c * hw
    _bn_relu_shuffle_kernel[(triton.cdiv(total, BLOCK),)](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
        OUT_C=out_c,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out[:, :c, :, :], out[:, c:, :, :]
