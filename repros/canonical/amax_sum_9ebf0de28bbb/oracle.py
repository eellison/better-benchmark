"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MT5 bidirectional relative-position attention softmax scope, including the logarithmic bucket lookup, bf16 embedding-bias gather, returned broadcast relative-bias tensor with its head-inner stride layout, Inductor-style fp32 score construction from bf16 operands, stable last-dimension amax/libdevice.exp/sum/div, final bf16 probability cast, expand, and contiguous returned view, whereas Inductor lowers the decomposed iota/abs/log/minimum/where/embedding/permute/zero-add/add/amax/sub/exp/sum/div/cast/expand/view graph through generic pointwise producers and reduction scheduling; Inductor cannot do this today because its pattern library does not canonicalize MT5 relative-position bucket construction plus an observable bias side output into one full-scope attention-softmax plan; the fix is NEW_PATTERN: add a guarded MT5 relative-position attention-softmax lowering that recomputes bucket indices at point of use, emits required side outputs in their requested layout, and sinks layout-only epilogues into the store schedule."""

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
def _relative_position_softmax_kernel(
    x_ptr,
    rel_bias_ptr,
    bias_out_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    head = flat_bh - batch * HEADS
    query = rows - flat_bh * Q_LEN
    row_offsets = rows[:, None] * K_LEN + cols[None, :]

    bucket = _t5_bidirectional_bucket(query[:, None], cols[None, :])
    bias = tl.load(
        rel_bias_ptr + bucket * HEADS + head[:, None],
        mask=active,
        other=0.0,
    )
    bias = (bias.to(tl.float32) + 0.0).to(tl.bfloat16)

    bias_offsets = (
        batch[:, None] * (HEADS * Q_LEN * K_LEN)
        + head[:, None]
        + query[:, None] * (K_LEN * HEADS)
        + cols[None, :] * HEADS
    )
    tl.store(bias_out_ptr + bias_offsets, bias, mask=active)

    x = tl.load(x_ptr + row_offsets, mask=active, other=0.0).to(tl.float32)
    scores = x + bias.to(tl.float32)
    scores = tl.where(active, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(out_ptr + row_offsets, probs, mask=active)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, _shape0, _shape1, _shape2, _shape3, _shape4 = inputs
    del _shape0, _shape1, _shape2, _shape3, _shape4

    batch = 32
    heads = 6
    q_len = 128
    k_len = 128
    rows = batch * heads * q_len

    relative_bias = torch.empty_strided(
        (batch, heads, q_len, k_len),
        (heads * q_len * k_len, 1, heads * k_len, heads),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        (batch * heads, q_len, k_len),
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _relative_position_softmax_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        relative_bias,
        out,
        ROWS=rows,
        HEADS=heads,
        Q_LEN=q_len,
        K_LEN=k_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return relative_bias, out


# e7595a1e: MT5 bidirectional relative-position attention softmax, B=32, H=6, S=128.
@oracle_impl(hardware="B200", point="e7595a1e", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
