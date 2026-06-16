"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo bf16 token embedding plus generated-position embedding LayerNorm scope in one Triton row kernel, including the indexed token gather, generated position gather, explicit bf16 token-plus-position add returned as `[32,128,2048]`, fp32 population var_mean over the rounded bf16 add, eps=1e-5 normalization, bf16 affine scale/bias epilogue, three aliasing `[4096,2048]` normalized views, and the deterministic all-false adjacent-position mask, whereas Inductor lowers the embedding producers, iota/expand/cat/slice mask construction, normalization, affine cast, and alias views through generic embedding, pointwise, and norm-template schedules; Inductor cannot do this today because norm-template canonicalization does not recognize this GPT-Neo generated-position embedding LayerNorm with sibling mask emission, bf16 rounding boundaries, and multi-view alias returns as one semantic pattern; the fix is NEW_PATTERN: add a guarded GPT-Neo embedding-LayerNorm lowering that folds token and position gathers, row statistics, affine stores, constant-mask epilogue, and alias-view returns into one scheduled region."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _gptneo_embedding_layernorm_mask_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    mask_out_ptr,
    norm_out_ptr,
    rows: tl.constexpr,
    seq: tl.constexpr,
    hidden: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < hidden
    offsets = row * hidden + cols
    seq_index = row - (row // seq) * seq

    token_id = tl.load(token_ids_ptr + row)
    token = tl.load(
        token_table_ptr + token_id * hidden + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + seq_index * hidden + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    added = (token + position).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(add_out_ptr + offsets, added, mask=col_mask)
    tl.store(mask_out_ptr + row, False)

    x = added.to(tl.float32).to(tl.float64)
    x_for_reduce = tl.where(col_mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=0) / hidden
    centered = x - mean
    centered_for_var = tl.where(col_mask, centered, 0.0)
    variance = tl.sum(centered_for_var * centered_for_var, axis=0) / hidden
    invstd = libdevice.rsqrt(variance + eps)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float64)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float64)
    affine = centered * invstd * weight + bias
    tl.store(norm_out_ptr + offsets, affine.to(tl.float32).to(tl.bfloat16), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="5247d883", BLOCK_H=2048, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        view_shape0,
        view_shape1,
        _expand_shape,
        view_shape2,
    ) = inputs
    del _expand_shape

    batch = int(token_ids.shape[0])
    seq = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq

    add_out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    mask = torch.empty_strided(
        (batch, seq),
        (seq, 1),
        device=token_table.device,
        dtype=torch.bool,
    )

    _gptneo_embedding_layernorm_mask_kernel[(rows,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        add_out,
        mask,
        norm_base,
        rows=rows,
        seq=seq,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        add_out,
        norm_base.view(_shape_tuple(view_shape0)),
        norm_base.view(_shape_tuple(view_shape1)),
        mask,
        norm_base.view(_shape_tuple(view_shape2)),
    )
