"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J bf16 token embedding gather and hidden-size-4096 LayerNorm scope in one Triton row kernel, including the visible raw embedding output, fp32 population var_mean over dim 2 with eps=1e-5 rsqrt, bf16 affine output cast, and four returned `[128,4096]` aliases of one normalized buffer, whereas Inductor lowers the embedding producer, normalization reduction, affine epilogue, dtype casts, and repeated view returns through generic schedules; Inductor cannot do this today because norm canonicalization does not recognize direct bf16 embedding-gather producers feeding fixed-K fp32 LayerNorm with both the raw embedding side output and alias-only normalized outputs; the fix is NEW_PATTERN: add an embedding-LayerNorm alias lowering that folds indexed embedding loads, fp32 row statistics, affine epilogue, bf16 output stores, and repeated view aliases into one specialized lowering."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 4096
EPS = 1.0e-5


@triton.jit
def _embedding_layernorm_alias_kernel(
    table_ptr,
    token_ids_ptr,
    weight_ptr,
    bias_ptr,
    embedding_out_ptr,
    norm_out_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS_
    col_mask = cols < HIDDEN_
    mask = row_mask[:, None] & col_mask[None, :]
    out_offsets = rows[:, None] * HIDDEN_ + cols[None, :]

    token_ids = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
    embedding = tl.load(
        table_ptr + token_ids[:, None] * HIDDEN_ + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    tl.store(embedding_out_ptr + out_offsets, embedding, mask=mask)

    x = embedding.to(tl.float32)
    mean_acc = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
    m2_acc = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
    weight_acc = tl.zeros([ROW_BLOCK, BLOCK_H], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    centered = x - mean[:, None]
    invstd = libdevice.rsqrt((m2 / 4096.0) + EPS_)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = centered * invstd[:, None]
    affine = (normalized * weight[None, :]) + bias[None, :]
    tl.store(norm_out_ptr + out_offsets, affine, mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 4553b7e1: (T([50400,4096], bf16), T([1,128], i64), T([4096], bf16), T([4096], bf16), S([128,4096]) x4)
@oracle_impl(hardware="B200", point="4553b7e1", BLOCK_H=4096, ROW_BLOCK=1, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    table, token_ids, weight, bias, shape0, shape1, shape2, shape3 = inputs
    embedding = torch.empty_strided(
        (1, ROWS, HIDDEN),
        (ROWS * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=torch.bfloat16,
    )
    normalized = torch.empty_strided(
        (1, ROWS, HIDDEN),
        (ROWS * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=torch.bfloat16,
    )
    _embedding_layernorm_alias_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        table,
        token_ids,
        weight,
        bias,
        embedding,
        normalized,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        EPS_=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        embedding,
        normalized.view(_as_shape(shape0)),
        normalized.view(_as_shape(shape1)),
        normalized.view(_as_shape(shape2)),
        normalized.view(_as_shape(shape3)),
    )
