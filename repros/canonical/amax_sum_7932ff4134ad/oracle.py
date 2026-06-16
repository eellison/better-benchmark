"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 T5 bidirectional relative-position attention softmax scope, including the logarithmic bucket lookup, bf16 embedding-bias gather, returned broadcast relative-bias tensor with its head-inner stride layout, bf16-rounded score add, fp32 stable last-dimension amax/libdevice.exp/sum/div, final bf16 probability cast, expand, and contiguous returned view, whereas Inductor lowers the decomposed iota/abs/log/minimum/where/embedding/permute/zero-add/add/amax/sub/exp/sum/div/cast/expand/view graph through generic pointwise producers and reduction scheduling; Inductor cannot do this today because its pattern library does not canonicalize T5 relative-position bucket construction plus an observable bias side output into one full-scope attention-softmax plan; the fix is NEW_PATTERN: add a guarded T5 relative-position attention-softmax lowering that recomputes bucket indices at point of use, emits required side outputs in their requested layout, and sinks layout-only epilogues into the store schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _t5_bidirectional_bucket(query, cols):
    rel_pos = cols - query
    distance = tl.abs(rel_pos)
    bucket = distance
    bucket = tl.where(distance >= 8, 8, bucket)
    bucket = tl.where(distance >= 12, 9, bucket)
    bucket = tl.where(distance >= 16, 10, bucket)
    bucket = tl.where(distance >= 23, 11, bucket)
    bucket = tl.where(distance >= 32, 12, bucket)
    bucket = tl.where(distance >= 46, 13, bucket)
    bucket = tl.where(distance >= 64, 14, bucket)
    bucket = tl.where(distance >= 91, 15, bucket)
    return bucket + tl.where(rel_pos > 0, 16, 0)


@triton.jit
def _materialize_relative_bias_kernel(
    rel_bias_ptr,
    out_ptr,
    total: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    head = offsets % heads
    tmp = offsets // heads
    key = tmp % k_len
    query = (tmp // k_len) % q_len

    bucket = _t5_bidirectional_bucket(query, key)
    bias = tl.load(rel_bias_ptr + bucket * heads + head, mask=mask, other=0.0)
    bias = (bias.to(tl.float32) + 0.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, bias, mask=mask)


@triton.jit
def _relative_position_softmax_kernel(
    x_ptr,
    rel_bias_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    head = flat_bh % heads
    query = rows - flat_bh * q_len
    offsets = rows[:, None] * k_len + cols[None, :]

    bucket = _t5_bidirectional_bucket(query[:, None], cols[None, :])
    bias = tl.load(
        rel_bias_ptr + bucket * heads + head[:, None],
        mask=mask,
        other=0.0,
    )
    bias = (bias.to(tl.float32) + 0.0).to(tl.bfloat16)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scores = tl.where(mask, x + bias.to(tl.float32), -float("inf"))

    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(out_ptr + offsets, probs, mask=mask)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, BIAS_BLOCK: int, num_warps: int):
    arg0_1, arg1_1, shape0, _shape1, _shape2, _shape3, shape4 = inputs
    del _shape1, _shape2, _shape3

    full_shape = tuple(int(dim) for dim in shape0)
    out_shape = tuple(int(dim) for dim in shape4)
    batch = int(full_shape[0])
    heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(batch * heads * q_len)

    relative_bias = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, 1, heads * k_len, heads),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _materialize_relative_bias_kernel[(triton.cdiv(relative_bias.numel(), BIAS_BLOCK),)](
        arg1_1,
        relative_bias,
        total=relative_bias.numel(),
        heads=heads,
        q_len=q_len,
        k_len=k_len,
        BLOCK=BIAS_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _relative_position_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        out,
        n_rows=n_rows,
        heads=heads,
        q_len=q_len,
        k_len=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return relative_bias, out


# ea4c0a34: (T([64,1024,1024], bf16), T([32,8], bf16), S([8,8,1024,1024]), ...)
@oracle_impl(hardware="B200", point="ea4c0a34", BLOCK_M=1, BLOCK_N=1024, BIAS_BLOCK=256, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    BIAS_BLOCK: int,
    num_warps: int,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BIAS_BLOCK=BIAS_BLOCK,
        num_warps=num_warps,
    )
