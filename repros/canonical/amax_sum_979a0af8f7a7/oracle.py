"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5/MT5 additive-bias bf16 softmax scope, including the shape-param view, fp32 score+bias construction from bf16 inputs, fp32 stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 probability cast, expand, and final contiguous view in one Triton row kernel, whereas Inductor lowers the decomposed view/add/cast/amax/sub/exp/sum/div/cast/expand/view graph through generic reduction scheduling; Inductor cannot do this today because its pattern library does not recognize this strided broadcast-head additive attention softmax as one reusable full-scope row template with the fused convert-to-fp32 producer and layout-only epilogue sunk into the store; the fix is NEW_PATTERN: add an Inductor lowering for additive-bias bf16 attention softmax that fuses score construction, stable row reductions, final cast, and output-view emission."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bf16_add_softmax_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s1: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    n_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads
    query = rows - flat_bh * q_len

    x_offsets = flat_bh[:, None] * x_s0 + query[:, None] * x_s1 + cols[None, :] * x_s2
    bias_offsets = (
        batch[:, None] * bias_s0
        + head[:, None] * bias_s1
        + query[:, None] * bias_s2
        + cols[None, :] * bias_s3
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = x + bias
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    out_offsets = flat_bh[:, None] * out_s0 + query[:, None] * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=mask)


def _contiguous_3d_stride(shape):
    return (shape[1] * shape[2], shape[2], 1)


# d3b915a9: (T([192,128,128], bf16), T([32,6,128,128], bf16, stride=(98304,1,768,6)), ...)
@oracle_impl(hardware="B200", point="d3b915a9", BLOCK_M=8, BLOCK_N=128, num_warps=4)
# 679762f8: (T([64,1024,1024], bf16), T([8,8,1024,1024], bf16, stride=(8388608,1,8192,8)), ...)
@oracle_impl(hardware="B200", point="679762f8", BLOCK_M=1, BLOCK_N=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    n_rows = int(arg0_1.shape[0] * q_len)

    _bf16_add_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        out,
        x_s0=arg0_1.stride(0),
        x_s1=arg0_1.stride(1),
        x_s2=arg0_1.stride(2),
        bias_s0=arg1_1.stride(0),
        bias_s1=arg1_1.stride(1),
        bias_s2=arg1_1.stride(2),
        bias_s3=arg1_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        heads=arg1_1.shape[1],
        q_len=q_len,
        k_len=k_len,
        n_rows=n_rows,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
