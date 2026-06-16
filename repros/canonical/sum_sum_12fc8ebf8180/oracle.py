"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DCGAN bf16 BatchNorm-backward/leaky-ReLU scope by preserving the bf16 rounding after the gated leaky producer, cooperatively reducing the two compatible per-channel summaries from that rounded value, finalizing the fp32 sum and scale-gradient vectors, and emitting the dependent bf16 dense gradient tensor in the same channel program; Inductor schedules the rounded producer, sibling reductions, vector finalization, and broadcast epilogue as generic reduction/pointwise work, so it misses a multi-output channel split-K plan that co-finalizes shared summaries and feeds the dense epilogue without extra launches and traffic; the fix is COOPERATIVE_SPLIT_K: add a cooperative channel split-K reduction template that preserves the bf16 cast boundary while fusing the full BN-backward epilogue and returned side outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 0.0001220703125


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
def _channel_reduce_epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    sum_out_ptr,
    dot_scaled_out_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < K_TOTAL
    n = k // HW
    hw = k - n * HW
    offsets = n * (C * HW) + c * HW + hw

    x0 = tl.load(arg0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    gate = tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg3_ptr + c).to(tl.float32)
    weight = tl.load(arg4_ptr + c).to(tl.float32)
    grad_weight = tl.load(arg5_ptr + c).to(tl.float32)

    leaky = _f32_mul(x0, 0.2)
    selected = tl.where(gate > 0.0, x0, leaky).to(tl.bfloat16).to(tl.float32)
    centered = _f32_sub(x2, mean)
    dot = _f32_mul(selected, centered)
    sum_value = tl.sum(tl.where(active, selected, 0.0), axis=0)
    dot_value = tl.sum(tl.where(active, dot, 0.0), axis=0)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_scaled_out_ptr + c, _f32_mul(dot_value, weight))

    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    weight_sq = _f32_mul(weight, weight)
    correction_scale = _f32_mul(dot_mean, weight_sq)
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(selected, correction)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    centered_grad = _f32_sub(after_correction, mean_term)
    output_scale = _f32_mul(weight, grad_weight)
    out = _f32_mul(centered_grad, output_scale)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="a564ddd4", BLOCK_K=8192, num_warps=8)
@oracle_impl(hardware="B200", point="49f9b4bd", BLOCK_K=2048, num_warps=8)
@oracle_impl(hardware="B200", point="15406f9b", BLOCK_K=512, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    num_warps: int,
):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    n, c, h, w = (int(dim) for dim in arg0.shape)
    hw = h * w
    k_total = n * hw

    sum_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dot_scaled_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    _channel_reduce_epilogue_kernel[(c,)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        sum_out,
        dot_scaled_out,
        dense_out,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        SCALE_VALUE=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return sum_out, dot_scaled_out, dense_out
