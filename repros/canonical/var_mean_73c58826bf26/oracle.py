"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 singleton-spatial GroupNorm-style affine ReLU scope in one shape-specialized Triton row kernel, including the bf16-to-fp32 input promotion, population `var_mean(..., correction=0, keepdim=True)` over each of 32 groups, eps=1e-5 `rsqrt`, bf16 scale/bias promotion, f32 ReLU, and final contiguous bf16 output store, whereas Inductor lowers the decomposed view/var_mean/sub/rsqrt/affine/cast/ReLU graph through its generic persistent normalization scheduler; Inductor cannot do this today because its pattern library does not recognize this fixed-32-group inference GroupNorm affine ReLU motif as a small-group template with stripped generic reduction bookkeeping and direct channel-affine indexing; the fix is NEW_PATTERN: add a guarded GroupNorm affine ReLU lowering for fixed group count and power-of-two group extents that keeps row statistics in registers and emits the bf16 output directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


NUM_GROUPS = 32
EPS = 1.0e-5


@triton.jit
def _groupnorm_affine_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL_GROUPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    NUM_GROUPS_CONST: tl.constexpr,
    HW_SIZE: tl.constexpr,
    GROUP_ELEMS: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    elems = tl.arange(0, BLOCK_K)[None, :]
    mask = (rows < TOTAL_GROUPS) & (elems < GROUP_ELEMS)
    offsets = rows * GROUP_ELEMS + elems

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_for_sum = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1)[:, None] / GROUP_ELEMS
    centered = x - mean
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(centered_for_var * centered_for_var, axis=1)[:, None] / GROUP_ELEMS
    invstd = libdevice.rsqrt(variance + EPSILON)

    group = rows % NUM_GROUPS_CONST
    channels_per_group: tl.constexpr = CHANNELS // NUM_GROUPS_CONST
    channel = group * channels_per_group + elems // HW_SIZE
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    normalized = centered * invstd
    affine = normalized * weight + bias
    zero = tl.full([ROW_BLOCK, BLOCK_K], 0, tl.int32)
    relu = triton_helpers.maximum(zero, affine)
    tl.store(out_ptr + offsets, relu, mask=mask)


@triton.jit
def _groupnorm_affine_relu_k64_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, 64)[None, :]

    offsets = r0_index + 64 * xindex
    x = tl.load(x_ptr + offsets, None, eviction_policy="evict_first").to(tl.float32)
    weight = tl.load(
        weight_ptr + (((r0_index + 64 * xindex) // 16) % 128),
        None,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + (((r0_index + 64 * xindex) // 16) % 128),
        None,
        eviction_policy="evict_last",
    ).to(tl.float32)

    tmp1 = x.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, 64])
    tmp3 = tmp2 * tmp2
    tmp6 = tl.sum(tmp2, 1)[:, None].to(tl.float32)
    tmp6_sq = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp7 = (tl.full([1, 1], 64, tl.int32)).to(tl.float32)
    tmp8 = tmp6 / tmp7
    tmp13 = tmp6_sq / tmp7 - tmp8 * tmp8
    tmp14 = tmp1 - tmp8
    tmp16 = triton_helpers.maximum(tl.full([1, 1], 0, tl.int32), tmp13)
    tmp17 = tl.full([1, 1], 1.0e-5, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = weight.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp25 = bias.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.full([1, 1], 0, tl.int32)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tl.store(out_ptr + offsets, tmp29, None)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 011e9762: (T([128,512,1,1], bf16), T([512], bf16), T([512], bf16), S([128,32,16,1]), S([128,512,1,1]))
@oracle_impl(hardware="B200", point="011e9762", BLOCK_K=16, ROW_BLOCK=16, num_warps=1, num_stages=3)
# e42af87f: (T([128,256,2,2], bf16), T([256], bf16), T([256], bf16), S([128,32,8,4]), S([128,256,2,2]))
@oracle_impl(hardware="B200", point="e42af87f", BLOCK_K=32, ROW_BLOCK=8, num_warps=1, num_stages=3)
# 4598cdde: (T([128,128,4,4], bf16), T([128], bf16), T([128], bf16), S([128,32,4,16]), S([128,128,4,4]))
@oracle_impl(hardware="B200", point="4598cdde", BLOCK_K=64, ROW_BLOCK=4, num_warps=1, num_stages=3)
# d0b6be6c: (T([128,64,8,8], bf16), T([64], bf16), T([64], bf16), S([128,32,2,64]), S([128,64,8,8]))
@oracle_impl(hardware="B200", point="d0b6be6c", BLOCK_K=128, ROW_BLOCK=2, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_K: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    shape = tuple(int(dim) for dim in arg0_1.shape)
    batch, channels, height, width = shape
    hw_size = height * width
    group_elems = channels // NUM_GROUPS * hw_size
    total_groups = batch * NUM_GROUPS

    out = torch.empty_strided(
        shape,
        _contiguous_stride(shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    if group_elems == 64 and channels == 128 and hw_size == 16:
        _groupnorm_affine_relu_k64_kernel[(triton.cdiv(total_groups, ROW_BLOCK),)](
            arg0_1,
            arg1_1,
            arg2_1,
            out,
            XBLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _groupnorm_affine_relu_kernel[(triton.cdiv(total_groups, ROW_BLOCK),)](
            arg0_1,
            arg1_1,
            arg2_1,
            out,
            TOTAL_GROUPS=total_groups,
            CHANNELS=channels,
            NUM_GROUPS_CONST=NUM_GROUPS,
            HW_SIZE=hw_size,
            GROUP_ELEMS=group_elems,
            EPSILON=EPS,
            BLOCK_K=BLOCK_K,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
