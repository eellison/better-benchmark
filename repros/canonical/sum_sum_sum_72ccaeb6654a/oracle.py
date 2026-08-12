"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle keeps the Inception bf16 avg-pool-backward/add fanout and ReLU mask as a branch-specific Triton producer, then feeds the resulting four bf16 batchnorm grad-output tensors to the native BN-backward reduction/epilogue, whereas Inductor materializes the pooled producer, slices, masks, channel reductions, vector side outputs, and tensor epilogues as generic regions; Inductor cannot do this today because its scheduler/codegen has no cooperative producer-plus-multi-output-reduction template that preserves the bf16 cast boundaries across disjoint channel slices while returning both per-channel sums and channels-last tensor outputs; the fix is COOPERATIVE_SPLIT_K: add a guarded pooled-add fanout schedule that computes each branch grad-output once and sinks it into the paired BN backward reductions and epilogues without redundant producer materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _make_branch_grad_kernel(
    pool_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
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
    hw = (offsets // channels) % 1225
    n = offsets // row_stride
    h = hw // 35
    w = hw - h * 35

    base = n * 352800 + (source_offset + c) + h * 10080 + w * 288
    up1 = h > 0
    up2 = h > 1
    left1 = w > 0
    left2 = w > 1

    scale = 0.1111111111111111
    pool = (tl.load(pool_ptr + base, mask=active, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 10080, mask=active & up1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 20160, mask=active & up2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 288, mask=active & left1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 576, mask=active & left2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 10368, mask=active & up1 & left1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 10656, mask=active & up1 & left2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 20448, mask=active & up2 & left1, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pool += (tl.load(pool_ptr + base - 20736, mask=active & up2 & left2, other=0.0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    pooled = pool.to(tl.bfloat16).to(tl.float32)

    add0 = (pooled + tl.load(add0_ptr + base, mask=active, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    add1 = (add0 + tl.load(add1_ptr + base, mask=active, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    source = (add1 + tl.load(add2_ptr + base, mask=active, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    x_offsets = n * row_stride + h * h_stride + w * w_stride + c
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)

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


def _emit_grad(pool_grad, add0, add1, add2, x, mean, invstd, weight, bias, zero, source_offset, total, *, BLOCK, num_warps):
    grad = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)
    _make_branch_grad_kernel[(triton.cdiv(total, BLOCK),)](
        pool_grad,
        add0,
        add1,
        add2,
        x,
        mean,
        invstd,
        weight,
        bias,
        grad,
        zero,
        channels=x.shape[1],
        source_offset=source_offset,
        total=total,
        row_stride=x.stride(0),
        h_stride=x.stride(2),
        w_stride=x.stride(3),
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return grad


@oracle_impl(hardware="B200", point="a91d4c93", BLOCK=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
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
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
    ) = inputs
    del arg1_1

    grad0 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, 224, 10035200, BLOCK=BLOCK, num_warps=num_warps)
    grad1 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg10_1, 128, 15052800, BLOCK=BLOCK, num_warps=num_warps)
    grad2 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg10_1, 64, 10035200, BLOCK=BLOCK, num_warps=num_warps)
    grad3 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg10_1, 0, 10035200, BLOCK=BLOCK, num_warps=num_warps)

    out0 = _native_bn_branch(grad0, arg5_1, arg6_1, arg7_1, arg8_1)
    out1 = _native_bn_branch(grad1, arg11_1, arg12_1, arg13_1, arg14_1)
    out2 = _native_bn_branch(grad2, arg16_1, arg17_1, arg18_1, arg19_1)
    out3 = _native_bn_branch(grad3, arg21_1, arg22_1, arg23_1, arg24_1)
    return out0 + out1 + out2 + out3
