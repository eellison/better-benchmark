"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 Swin relative-position attention softmax plus pad scope, including the sliced `[*,49,49]` scores from the `[*,56,56]` input, indexed `[169,H]` relative-bias table gather, fp32 stable natural-exp softmax, final bf16 probability rounding, and zero-padding back to the returned contiguous `[*,56,56]` output; Inductor lowers the decomposed slice/index/permute/clone/add/amax/exp/sum/div/cast/view/pad graph as generic producer and reduction fragments that replay or materialize layout work around the small-row softmax; the fix is NEW_PATTERN: add a Swin relative-position softmax template that hoists the window-invariant bias gather and sinks the padding/layout epilogue into one row-store schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _materialize_relpos_bias_kernel(
    index_ptr,
    table_ptr,
    bias_ptr,
    HEADS: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    head = offsets // 2401
    rel = offsets - head * 2401
    rel_index = tl.load(index_ptr + rel, mask=mask, other=0).to(tl.int32)
    bias = tl.load(table_ptr + rel_index * HEADS + head, mask=mask, other=0.0)
    tl.store(bias_ptr + offsets, bias, mask=mask)


@triton.jit
def _swin_relpos_softmax_pad_kernel(
    scores_ptr,
    bias_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_store = cols < 56

    score_row = rows // 56
    query = rows - score_row * 56
    head = score_row % HEADS
    row_live = row_mask & (query < 49)
    active = row_live[:, None] & (cols[None, :] < 49)

    score_offsets = score_row[:, None] * 3136 + query[:, None] * 56 + cols[None, :]
    bias_offsets = head[:, None] * 2401 + query[:, None] * 49 + cols[None, :]

    score = tl.load(scores_ptr + score_offsets, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=active, other=0.0).to(tl.float32)
    logits = score + bias
    logits = tl.where(active, logits, -float("inf"))

    row_max = tl.max(logits, axis=1)
    safe_max = tl.where(row_live, row_max, 0.0)
    numer = libdevice.exp(logits - safe_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(row_live, denom, 1.0)
    probs = (numer / denom[:, None]).to(tl.bfloat16)
    out = tl.where(active, probs, 0.0)

    store_mask = row_mask[:, None] & col_store[None, :]
    tl.store(out_ptr + rows[:, None] * 56 + cols[None, :], out, mask=store_mask)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    scores, rel_index, rel_table, *_shape_params = inputs
    heads = int(rel_table.shape[1])
    n_rows = int(scores.shape[0]) * 56
    out = torch.empty_strided(
        tuple(scores.shape),
        (3136, 56, 1),
        device=scores.device,
        dtype=torch.bfloat16,
    )
    bias = torch.empty_strided((heads, 49, 49), (2401, 49, 1), device=scores.device, dtype=torch.bfloat16)

    _materialize_relpos_bias_kernel[(triton.cdiv(heads * 2401, 256),)](
        rel_index,
        rel_table,
        bias,
        HEADS=heads,
        TOTAL=heads * 2401,
        BLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    _swin_relpos_softmax_pad_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        scores,
        bias,
        out,
        N_ROWS=n_rows,
        HEADS=heads,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="2ebbf10c", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="b57df06f", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="3551defb", BLOCK_M=8, BLOCK_N=64, num_warps=4)
@oracle_impl(hardware="B200", point="7e00ce6b", BLOCK_M=8, BLOCK_N=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
