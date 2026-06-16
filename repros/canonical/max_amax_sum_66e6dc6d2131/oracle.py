"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-OSS bf16 softmax-with-extra-column scope by folding the view, scale, broadcast bias add, channel-bias `cat`, `max.dim` value path, bf16 subtract, fp32 stable amax/libdevice.exp/sum/div, final bf16 cast, slice, clone, expand, and returned contiguous view into one Triton row kernel, whereas Inductor lowers the captured scale/add/cat/max/sub/amax/exp/sum/div/cast/slice/clone/view graph through generic pointwise, reduction, and layout-copy regions; Inductor cannot do this today because its pattern library does not recognize this softmax over a virtual concatenated sentinel column with an observable bf16 max/sub boundary and a sliced materialized output as one full-scope row template; the fix is NEW_PATTERN: add a guarded virtual-cat softmax lowering that preserves the bf16 rounding boundaries, includes the extra per-channel column in the denominator, and sinks the slice/clone/view epilogue into the final store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _virtual_cat_softmax_kernel(
    x_ptr,
    bias_ptr,
    extra_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    CHANNELS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    CAT_LEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    cols = tl.arange(0, BLOCK_N)

    channel_mask = channels < CHANNELS
    data_col_mask = cols < K_LEN
    valid_col_mask = cols < CAT_LEN
    data_mask = channel_mask[:, None] & data_col_mask[None, :]
    valid_mask = channel_mask[:, None] & valid_col_mask[None, :]

    x_offsets = (
        channels[:, None] * x_s0
        + query * x_s1
        + cols[None, :] * x_s2
    )
    bias_offsets = query * bias_s2 + cols * bias_s3

    x = tl.load(x_ptr + x_offsets, mask=data_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=data_col_mask, other=0.0).to(tl.float32)
    scaled = _round_bf16_to_fp32(x * 0.125)
    data_scores = _round_bf16_to_fp32(scaled + bias[None, :])

    extra = tl.load(extra_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    scores = tl.where(cols[None, :] < K_LEN, data_scores, extra[:, None])
    scores = tl.where(valid_mask, scores, -float("inf"))

    first_max = tl.max(scores, axis=1)
    shifted_bf16 = _round_bf16_to_fp32(scores - first_max[:, None])
    shifted_bf16 = tl.where(valid_mask, shifted_bf16, -float("inf"))

    second_max = tl.max(shifted_bf16, axis=1)
    numer = libdevice.exp(shifted_bf16 - second_max[:, None])
    numer = tl.where(valid_mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    out_offsets = (
        channels[:, None] * out_s0
        + query * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=data_mask)


def _contiguous_3d_stride(shape):
    return (shape[1] * shape[2], shape[2], 1)


# 1bcb4709: (T([64,1000,1000], bf16), T([1,1,1000,1000], bf16), T([64], bf16), ...)
@oracle_impl(hardware="B200", point="1bcb4709", BLOCK_C=2, BLOCK_N=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, shape3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in shape3)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    channels = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])

    _virtual_cat_softmax_kernel[(q_len, triton.cdiv(channels, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        x_s0=arg0_1.stride(0),
        x_s1=arg0_1.stride(1),
        x_s2=arg0_1.stride(2),
        bias_s2=arg1_1.stride(2),
        bias_s3=arg1_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        CHANNELS=channels,
        Q_LEN=q_len,
        K_LEN=k_len,
        CAT_LEN=k_len + 1,
        BLOCK_C=BLOCK_C,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
