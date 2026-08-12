"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet weight-standardization backward tail in one Triton channel kernel, including the bf16-to-f32 input cast, clone/view logical row order, both per-channel sums, the returned scaled reduction, and the dependent full-tensor epilogue, whereas Inductor schedules the clone/view producers, sibling reductions, broadcast scalar math, and materializing epilogue as separate generic regions; Inductor cannot do this today because its scheduler does not form a full-scope multi-output reduction whose shared reduction results feed both a vector side output and a full tensor output through layout-changing clone/view inputs; the fix is SCHEDULER_FUSION: teach scheduler/codegen to fuse compatible sibling row reductions with all dependent broadcast epilogues and returned stores across simple clone/view layout producers."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _nfnet_weight_standardization_bwd_kernel(
    grad_ptr,
    weight_ptr,
    mean_ptr,
    invstd_ptr,
    gain_ptr,
    out_gain_ptr,
    out_weight_ptr,
    grad_s0: tl.constexpr,
    grad_s1: tl.constexpr,
    grad_s2: tl.constexpr,
    grad_s3: tl.constexpr,
    weight_s0: tl.constexpr,
    weight_s1: tl.constexpr,
    weight_s2: tl.constexpr,
    weight_s3: tl.constexpr,
    REDUCE_N: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    INV_REDUCE_N: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    oc = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    mask = k < REDUCE_N

    ic = k // 9
    rem = k - ic * 9
    kh = rem // 3
    kw = rem - kh * 3

    grad_offsets = oc * grad_s0 + ic * grad_s1 + kh * grad_s2 + kw * grad_s3
    weight_offsets = (
        oc * weight_s0 + ic * weight_s1 + kh * weight_s2 + kw * weight_s3
    )

    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + oc).to(tl.float32)
    invstd = tl.load(invstd_ptr + oc).to(tl.float32)
    gain = tl.load(gain_ptr + oc).to(tl.float32)

    centered = weight - mean
    sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)
    sum_grad_centered = tl.sum(tl.where(mask, grad * centered, 0.0), axis=0)

    mean_grad = sum_grad * INV_REDUCE_N
    invstd_sq = invstd * invstd
    variance_term = (sum_grad_centered * INV_REDUCE_N) * invstd_sq
    output_scale = invstd * (gain * WEIGHT_SCALE)

    tensor_out = (grad - centered * variance_term - mean_grad) * output_scale
    gain_out = (sum_grad_centered * invstd) * WEIGHT_SCALE

    tl.store(out_gain_ptr + oc, gain_out)
    tl.store(out_weight_ptr + oc * REDUCE_N + k, tensor_out, mask=mask)


# dbe7e0fa: (T([16,3,3,3], bf16, stride=(27,1,9,3)), T([16,3,3,3], f32, stride=(27,1,9,3)), ...)
# a7e28603: (T([32,16,3,3], bf16, stride=(144,1,48,16)), T([32,16,3,3], f32, stride=(144,1,48,16)), ...)
# 64510e66: (T([64,32,3,3], bf16, stride=(288,1,96,32)), T([64,32,3,3], f32, stride=(288,1,96,32)), ...)
# 3005b63c: (T([128,64,3,3], bf16, stride=(576,1,192,64)), T([128,64,3,3], f32, stride=(576,1,192,64)), ...)
# 442ee97d: (T([128,128,3,3], bf16, stride=(1152,1,384,128)), T([128,128,3,3], f32, stride=(1152,1,384,128)), ...)
# 8b771129: (T([256,128,3,3], bf16, stride=(1152,1,384,128)), T([256,128,3,3], f32, stride=(1152,1,384,128)), ...)
# a1b0079b: (T([768,128,3,3], bf16, stride=(1152,1,384,128)), T([768,128,3,3], f32, stride=(1152,1,384,128)), ...)
# 77446b6f: (T([64,64,3,3], bf16, stride=(576,1,192,64)), T([64,64,3,3], f32, stride=(576,1,192,64)), ...)
# f88f9f06: (T([384,64,3,3], bf16, stride=(576,1,192,64)), T([384,64,3,3], f32, stride=(576,1,192,64)), ...)
@oracle_impl(hardware="B200", point="dbe7e0fa", BLOCK=32, num_warps=1)
@oracle_impl(hardware="B200", point="a7e28603", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="64510e66", BLOCK=512, num_warps=4)
@oracle_impl(hardware="B200", point="3005b63c", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="442ee97d", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="8b771129", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="a1b0079b", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="77446b6f", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f88f9f06", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (
        grad,
        weight,
        mean,
        invstd,
        gain,
        _shape_param_0,
        _shape_param_1,
        out_gain_shape,
        out_weight_shape,
    ) = inputs

    out_channels = int(grad.shape[0])
    in_channels = int(grad.shape[1])
    reduce_n = in_channels * 9

    out_gain_shape = tuple(int(dim) for dim in out_gain_shape)
    out_weight_shape = tuple(int(dim) for dim in out_weight_shape)
    out_gain = torch.empty_strided(
        out_gain_shape,
        (1, 1, 1, 1),
        device=grad.device,
        dtype=torch.float32,
    )
    out_weight = torch.empty_strided(
        out_weight_shape,
        (reduce_n, 9, 3, 1),
        device=grad.device,
        dtype=torch.float32,
    )

    _nfnet_weight_standardization_bwd_kernel[(out_channels,)](
        grad,
        weight,
        mean,
        invstd,
        gain,
        out_gain,
        out_weight,
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        grad.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        REDUCE_N=reduce_n,
        IN_CHANNELS=in_channels,
        INV_REDUCE_N=0.037037037037037035,
        WEIGHT_SCALE=0.19245008972987526,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out_gain, out_weight
