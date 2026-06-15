"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT bf16 residual LayerNorm fragment in one fixed-hidden row-reduction kernel, including the visible bf16 residual-add output, captured bf16 mean/sub/mul/sqrt/div/add rounding boundaries, unbiased fp32 variance over the rounded residual add, and three aliasing `[2048, 768]` view returns, whereas Inductor lowers the returned residual producer, mean/variance row reductions, affine epilogue, and alias-only views through generic normalization scheduling; Inductor cannot do this today because the row-normalization scheduler does not keep a live visible residual producer and all dependent row-statistics/affine/view outputs in one full-scope template while preserving explicit dtype boundaries; the fix is SCHEDULER_FUSION: teach the fixed-hidden normalization template to fuse visible residual producers, bf16 rounded epilogues, and alias-view materialization in one guarded schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
        triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
    ],
    key=["ROWS", "HIDDEN"],
)
@triton.jit
def _bert_residual_layernorm_alias_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    added = (residual + flat).to(tl.bfloat16).to(tl.float32)
    tl.store(add_out_ptr + offsets, added.to(tl.bfloat16), mask=mask)

    reduce_input = tl.where(mask, added, 0.0)
    row_sum = tl.sum(reduce_input, axis=1)[:, None]
    mean_f32 = row_sum / HIDDEN
    mean = mean_f32.to(tl.bfloat16).to(tl.float32)
    centered = (added - mean).to(tl.bfloat16).to(tl.float32)

    sum_x2 = tl.sum(tl.where(mask, added * added, 0.0), axis=1)[:, None]
    variance = (sum_x2 - row_sum * mean_f32) / (HIDDEN - 1.0)
    std = tl.sqrt(tl.maximum(variance, 0.0)).to(tl.bfloat16).to(tl.float32)

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
    scaled = (weight[None, :] * centered).to(tl.bfloat16).to(tl.float32)
    denom = (std + 1.0e-6).to(tl.bfloat16).to(tl.float32)
    divided = (scaled / denom).to(tl.bfloat16).to(tl.float32)
    out = (divided + bias[None, :]).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, out, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="1c404995",
    BLOCK_H=1024,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    del shape2, shape3

    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = _shape_tuple(shape0)
    out_shape = _shape_tuple(shape1)
    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    norm_out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    grid = lambda meta: (triton.cdiv(rows, meta["ROW_BLOCK"]),)
    _bert_residual_layernorm_alias_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
    )
    return add_out, norm_out, norm_out, norm_out
