"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete grouped BN-backward/ReLU-mask scope by streaming the two bf16 activation tensors once per `(batch, channel group)`, preserving the bf16-to-fp32 input casts, keeping `sum(where)` and `sum(where * x1)` in registers, and deriving both f32 channel vectors plus the bf16 full-tensor epilogue from those summaries, whereas Inductor schedules the casts, view-threaded mask producer, sibling spatial reductions, dependent grouped reductions, full tensor epilogue, and final channel reductions as separate generic regions over materialized intermediates; Inductor cannot do this today because its algebraic simplifier does not recognize this fixed 32-group BN-backward reduction chain or keep the two base spatial summaries live across all consumers; the fix is ALGEBRAIC_ELIMINATION: add a guarded rewrite for this grouped BN-backward-style pattern that lowers the shared spatial summaries and dependent vector/full-tensor epilogues as one fused multi-output reduction plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


GROUPS = 32
GROUP_SCALE = 0.0078125


@triton.jit
def _zero_vectors_kernel(
    out_vec_ptr,
    out_sum_ptr,
    c_total: tl.constexpr,
    block_c: tl.constexpr,
):
    offsets = tl.arange(0, block_c)
    mask = offsets < c_total
    zeros = tl.zeros((block_c,), dtype=tl.float32)
    tl.store(out_vec_ptr + offsets, zeros, mask=mask)
    tl.store(out_sum_ptr + offsets, zeros, mask=mask)


@triton.jit
def _grouped_bn_backward_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    out_vec_ptr,
    out_sum_ptr,
    out_full_ptr,
    arg0_s0: tl.constexpr,
    arg0_s1: tl.constexpr,
    arg0_s2: tl.constexpr,
    arg0_s3: tl.constexpr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    arg2_s0: tl.constexpr,
    arg2_s1: tl.constexpr,
    arg3_s0: tl.constexpr,
    arg3_s1: tl.constexpr,
    c_total: tl.constexpr,
    groups: tl.constexpr,
    group_width: tl.constexpr,
    group_scale: tl.constexpr,
    hw_size: tl.constexpr,
    width: tl.constexpr,
    block_group: tl.constexpr,
    block_hw: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // groups
    group = pid - n * groups

    r = tl.arange(0, block_group)
    hw = tl.arange(0, block_hw)
    channel = group * group_width + r
    h = hw // width
    w = hw - h * width

    input0_offsets = (
        n * arg0_s0
        + channel[:, None] * arg0_s1
        + h[None, :] * arg0_s2
        + w[None, :] * arg0_s3
    )
    input1_offsets = (
        n * arg1_s0
        + channel[:, None] * arg1_s1
        + h[None, :] * arg1_s2
        + w[None, :] * arg1_s3
    )

    x0 = tl.load(arg0_ptr + input0_offsets).to(tl.float32)
    x1 = tl.load(arg1_ptr + input1_offsets).to(tl.float32)
    mean = tl.load(arg2_ptr + n * arg2_s0 + group * arg2_s1).to(tl.float32)
    invstd = tl.load(arg3_ptr + n * arg3_s0 + group * arg3_s1).to(tl.float32)
    gamma = tl.load(arg4_ptr + channel).to(tl.float32)
    beta = tl.load(arg5_ptr + channel).to(tl.float32)
    masked_value = tl.load(arg6_ptr).to(tl.float32)

    normalized = (x1 - mean) * invstd
    relu_input = normalized * gamma[:, None] + beta[:, None]
    where_value = tl.where(relu_input <= 0.0, masked_value, x0)

    sum_mul = tl.sum(where_value * x1, axis=1)
    sum_where = tl.sum(where_value, axis=1)

    grouped_mul = tl.sum(sum_mul * gamma, axis=0)
    grouped_where = tl.sum(sum_where * gamma, axis=0)

    grouped_scale = (grouped_where * mean - grouped_mul) * invstd * invstd * invstd
    grouped_scale = grouped_scale * group_scale
    grouped_bias = -(grouped_scale * mean) - grouped_where * invstd * group_scale
    out_full = where_value * (invstd * gamma)[:, None] + x1 * grouped_scale + grouped_bias

    output_offsets = n * c_total * hw_size + channel[:, None] * hw_size + hw[None, :]
    tl.store(out_full_ptr + output_offsets, out_full.to(tl.bfloat16))

    out_vec_contrib = (sum_mul - sum_where * mean) * invstd
    tl.atomic_add(out_vec_ptr + channel, out_vec_contrib, sem="relaxed")
    tl.atomic_add(out_sum_ptr + channel, sum_where, sem="relaxed")


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# e311375b: (T([128,64,8,8], bf16, channels-last-like), T([128,64,8,8], bf16, channels-last-like), ...)
@oracle_impl(hardware="B200", point="e311375b", group_width=2, hw_size=64, width=8, num_warps=2)
# c9ab4bb9: (T([128,128,4,4], bf16, channels-last-like), T([128,128,4,4], bf16, channels-last-like), ...)
@oracle_impl(hardware="B200", point="c9ab4bb9", group_width=4, hw_size=16, width=4, num_warps=2)
# eeb75e7f: (T([128,256,2,2], bf16, channels-last-like), T([128,256,2,2], bf16, channels-last-like), ...)
@oracle_impl(hardware="B200", point="eeb75e7f", group_width=8, hw_size=4, width=2, num_warps=1)
# 13564432: (T([128,512,1,1], bf16), T([128,512,1,1], bf16), ...)
@oracle_impl(hardware="B200", point="13564432", group_width=16, hw_size=1, width=1, num_warps=1)
def oracle_forward(inputs, *, group_width: int, hw_size: int, width: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        _arg4_1,
        _arg5_1,
        _arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        full_shape_arg,
        _shape_param_9,
        _shape_param_10,
        vec_shape_arg,
    ) = inputs

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    full_shape = tuple(int(dim) for dim in full_shape_arg)
    vec_shape = tuple(int(dim) for dim in vec_shape_arg)

    out_vec = torch.empty_strided(vec_shape, (1,), device=arg0_1.device, dtype=torch.float32)
    out_sum = torch.empty_strided(vec_shape, (1,), device=arg0_1.device, dtype=torch.float32)
    out_full = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    zero_block = triton.next_power_of_2(channels)
    _zero_vectors_kernel[(1,)](
        out_vec,
        out_sum,
        c_total=channels,
        block_c=zero_block,
        num_warps=4,
    )
    _grouped_bn_backward_kernel[(batch * GROUPS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        _arg4_1,
        _arg5_1,
        _arg6_1,
        out_vec,
        out_sum,
        out_full,
        arg0_s0=arg0_1.stride(0),
        arg0_s1=arg0_1.stride(1),
        arg0_s2=arg0_1.stride(2),
        arg0_s3=arg0_1.stride(3),
        arg1_s0=arg1_1.stride(0),
        arg1_s1=arg1_1.stride(1),
        arg1_s2=arg1_1.stride(2),
        arg1_s3=arg1_1.stride(3),
        arg2_s0=arg2_1.stride(0),
        arg2_s1=arg2_1.stride(1),
        arg3_s0=arg3_1.stride(0),
        arg3_s1=arg3_1.stride(1),
        c_total=channels,
        groups=GROUPS,
        group_width=group_width,
        group_scale=GROUP_SCALE,
        hw_size=hw_size,
        width=width,
        block_group=group_width,
        block_hw=hw_size,
        num_warps=num_warps,
        num_stages=3,
    )
    return out_vec, out_sum, out_full
