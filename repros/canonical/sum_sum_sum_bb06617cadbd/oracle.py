"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the returned bf16 Inception avg-pool-backward/add fanout once with a Triton producer, emits the two ReLU-gated BN-backward branch inputs from that producer, then feeds those branch grad-output tensors to native BN backward for the exact channel reductions, scale-gradient vectors, and channels-last bf16 tensor epilogues; Inductor currently schedules the avg-pool backward, three bf16 add stages, returned full producer, two channel slices, ReLU masks, four channel reductions, and dependent tensor epilogues as generic materialized regions; Inductor cannot do this today because its scheduler does not form one dependent multi-output plan that preserves bf16 cast boundaries while sharing a structured pooled producer between a returned side output and sibling BN-backward reductions; the fix is SCHEDULER_FUSION: add a guarded full-scope pooled-add fanout schedule that emits the returned producer once and sinks branch views into the paired BN reductions and epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add2_kernel(
    pool_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    out_add2_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < 28409856
    hw = (offsets // 768) % 289
    h = hw // 17
    w = hw - h * 17

    up1 = h > 0
    up2 = h > 1
    left1 = w > 0
    left2 = w > 1

    scale = 0.1111111111111111
    pool = (tl.load(pool_ptr + offsets, mask=active, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 13056, mask=active & up1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 26112, mask=active & up2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 768, mask=active & left1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 1536, mask=active & left2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 13824, mask=active & up1 & left1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 14592, mask=active & up1 & left2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 26880, mask=active & up2 & left1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + offsets - 27648, mask=active & up2 & left2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pooled = pool.to(tl.bfloat16).to(tl.float32)

    add_stage0 = (pooled + tl.load(add0_ptr + offsets, mask=active, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    add_stage1 = (add_stage0 + tl.load(add1_ptr + offsets, mask=active, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    source = (add_stage1 + tl.load(add2_ptr + offsets, mask=active, other=0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(out_add2_ptr + offsets, source, mask=active)


@triton.jit
def _branch_grad_from_add2_kernel(
    add2_full_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    grad_ptr,
    zero_ptr,
    channels: tl.constexpr,
    source_offset: tl.constexpr,
    total: tl.constexpr,
    row_stride: tl.constexpr,
    h_stride: tl.constexpr,
    w_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total
    c = offsets % channels
    hw = (offsets // channels) % 289
    n = offsets // row_stride
    h = hw // 17
    w = hw - h * 17
    x_offsets = n * row_stride + h * h_stride + w * w_stride + c
    full_offsets = n * 221952 + h * 13056 + w * 768 + source_offset + c

    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    source = tl.load(add2_full_ptr + full_offsets, mask=active, other=0.0).to(tl.float32)

    affine = (x - mean) * invstd * weight + bias
    grad = tl.where(affine.to(tl.bfloat16).to(tl.float32) <= 0.0, tl.load(zero_ptr).to(tl.float32), source).to(tl.bfloat16)
    tl.store(grad_ptr + x_offsets, grad, mask=active)


def _native_bn_branch(grad, x, mean, invstd, weight):
    grad_input, grad_weight, grad_bias = torch.ops.aten.native_batch_norm_backward.default(
        grad,
        x,
        weight,
        None,
        None,
        mean.reshape(-1),
        invstd.reshape(-1),
        True,
        0.0,
        [True, True, True],
    )
    return grad_bias, grad_weight, grad_input


@oracle_impl(hardware="B200", point="0d731111", ADD_BLOCK=1024, BRANCH_BLOCK=1024, ADD_WARPS=8, BRANCH_WARPS=4)
def oracle_forward(inputs, *, ADD_BLOCK: int, BRANCH_BLOCK: int, ADD_WARPS: int, BRANCH_WARPS: int):
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
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
    ) = inputs
    del arg1_1

    out_add2 = torch.empty_strided(tuple(arg0_1.shape), tuple(arg0_1.stride()), device=arg0_1.device, dtype=torch.bfloat16)
    grad96 = torch.empty_strided(tuple(arg5_1.shape), tuple(arg5_1.stride()), device=arg5_1.device, dtype=torch.bfloat16)
    grad384 = torch.empty_strided(tuple(arg11_1.shape), tuple(arg11_1.stride()), device=arg11_1.device, dtype=torch.bfloat16)

    _add2_kernel[(triton.cdiv(28409856, ADD_BLOCK),)](
        arg0_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out_add2,
        BLOCK=ADD_BLOCK,
        num_warps=ADD_WARPS,
        num_stages=4,
    )
    _branch_grad_from_add2_kernel[(triton.cdiv(3551232, BRANCH_BLOCK),)](
        out_add2,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        grad96,
        arg10_1,
        channels=96,
        source_offset=384,
        total=3551232,
        row_stride=arg5_1.stride(0),
        h_stride=arg5_1.stride(2),
        w_stride=arg5_1.stride(3),
        BLOCK=BRANCH_BLOCK,
        num_warps=BRANCH_WARPS,
        num_stages=4,
    )
    _branch_grad_from_add2_kernel[(triton.cdiv(14204928, BRANCH_BLOCK),)](
        out_add2,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        grad384,
        arg10_1,
        channels=384,
        source_offset=0,
        total=14204928,
        row_stride=arg11_1.stride(0),
        h_stride=arg11_1.stride(2),
        w_stride=arg11_1.stride(3),
        BLOCK=BRANCH_BLOCK,
        num_warps=BRANCH_WARPS,
        num_stages=4,
    )

    out96 = _native_bn_branch(grad96, arg5_1, arg6_1, arg7_1, arg8_1)
    out384 = _native_bn_branch(grad384, arg11_1, arg12_1, arg13_1, arg14_1)
    return (out_add2,) + out96 + out384
