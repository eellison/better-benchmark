"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-OSS generated-causal-mask virtual-column softmax scope in one Triton row kernel, including the visible bf16 `[1,1,1000,1000]` triangular `where` mask, `[64,1000,1000] -> [1,64,1000,1000]` metadata view, score scaling by 0.125, causal mask add, per-channel extra-column `cat`, `max.dim` value path, stable amax/libdevice.exp/sum/div, final bf16 cast, slice, clone, expand, and returned contiguous `[64,1000,1000]` view, whereas Inductor lowers the iota/unsqueeze/le/where mask construction and the virtual-column softmax as generic pointwise, reduction, and layout-copy regions; Inductor cannot do this today because its pattern library does not recognize this generated causal mask feeding a softmax over a virtual per-channel sentinel column while also emitting the observable mask output and preserving the compiled softmax numerics envelope; the fix is NEW_PATTERN: add a guarded generated-causal-mask virtual-cat softmax lowering that recomputes the structured predicate inside the row-softmax plan, stores the mask side output, includes the extra column in the denominator, and sinks the slice/clone/view epilogue into the final store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _causal_virtual_cat_softmax_kernel(
    x_ptr,
    extra_ptr,
    where_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
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

    keep = cols <= query
    mask_bias = tl.where(keep, 0.0, -3.3895313892515355e38)
    tl.store(
        where_ptr + query * K_LEN + cols,
        mask_bias,
        mask=(channel_block == 0) & data_col_mask,
    )

    x_offsets = (
        channels[:, None] * x_s0
        + query * x_s1
        + cols[None, :] * x_s2
    )
    x = tl.load(x_ptr + x_offsets, mask=data_mask, other=0.0).to(tl.float32)
    data_scores = x * 0.125 + mask_bias[None, :]

    extra = tl.load(extra_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    scores = tl.where(cols[None, :] < K_LEN, data_scores, extra[:, None])
    scores = tl.where(valid_mask, scores, -float("inf"))

    first_max = tl.max(scores, axis=1)
    shifted = scores - first_max[:, None]
    shifted = tl.where(valid_mask, shifted, -float("inf"))

    second_max = tl.max(shifted, axis=1)
    numer = libdevice.exp(shifted - second_max[:, None])
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
    return (int(shape[1]) * int(shape[2]), int(shape[2]), 1)


# d9b19a63: GPT-OSS bf16 generated causal mask plus virtual extra-column softmax.
@oracle_impl(hardware="B200", point="d9b19a63", BLOCK_C=2, BLOCK_N=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, shape4 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    channels = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    out_shape = tuple(int(dim) for dim in shape4)

    where = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _causal_virtual_cat_softmax_kernel[(q_len, triton.cdiv(channels, BLOCK_C))](
        arg0_1,
        arg1_1,
        where,
        out,
        x_s0=arg0_1.stride(0),
        x_s1=arg0_1.stride(1),
        x_s2=arg0_1.stride(2),
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
    return where, out
