"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J scaled additive attention softmax scope, including the shape-param view, fp32 divide-by-16, bf16 bias promotion, stable last-dimension amax/exp/sum/div, explicit bf16 rounding, expand, and final contiguous view in one Triton row kernel, whereas Inductor lowers the decomposed view/div/add/amax/sub/exp/sum/div/cast/expand/view graph as generic reduction and pointwise work; Inductor cannot do this today because its pattern library does not recognize this scaled bias-add returned bf16 softmax as one semantic row template with the layout-only epilogue sunk into the store; the fix is NEW_PATTERN: add an Inductor lowering that fuses scaled score construction, stable row-softmax reductions, bf16 cast, and final view emission into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _scaled_bias_softmax_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    n_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    heads = rows // q_len
    queries = rows - heads * q_len
    x_offsets = heads[:, None] * x_s0 + queries[:, None] * x_s1 + cols[None, :] * x_s2
    bias_offsets = queries[:, None] * bias_s2 + cols[None, :] * bias_s3

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = x * 0.0625 + bias
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    out_offsets = heads[:, None] * out_s0 + queries[:, None] * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=mask)


def _contiguous_3d_stride(shape):
    return (shape[1] * shape[2], shape[2], 1)


# 030f75c1: (T([16, 128, 128], f32), T([1, 1, 128, 128], bf16), ...)
@oracle_impl(hardware="B200", point="030f75c1", BLOCK_M=4, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _scaled_bias_softmax_kernel[(triton.cdiv(arg0_1.shape[0] * arg0_1.shape[1], BLOCK_M),)](
        arg0_1,
        arg1_1,
        out,
        q_len=arg0_1.shape[1],
        k_len=arg0_1.shape[2],
        x_s0=arg0_1.stride(0),
        x_s1=arg0_1.stride(1),
        x_s2=arg0_1.stride(2),
        bias_s2=arg1_1.stride(2),
        bias_s3=arg1_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        n_rows=arg0_1.shape[0] * arg0_1.shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
