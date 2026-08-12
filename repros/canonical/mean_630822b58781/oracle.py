"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 T5/MT5 token embedding gather, raw contiguous embedding side output, fp32 hidden-size-512 RMS mean(square)+eps=1e-6 rsqrt, required bf16 normalization round, bf16 affine weight multiply, and three returned flattened alias views from one normalized storage in one shape-specialized Triton row-block kernel, whereas Inductor lowers the embedding producer, generic mean reduction, RMSNorm epilogue, raw side output, and sibling alias returns through separate general scheduling machinery; Inductor cannot do this today because norm-template canonicalization does not recognize an indexed token embedding feeding fixed-hidden RMSNorm while preserving the raw embedding side output and repeated view aliases from one normalized storage; the fix is NEW_PATTERN: add an embedding-RMSNorm alias template that folds indexed embedding loads, raw side-output stores, fp32 row RMS reduction, bf16 epilogue rounding, and alias-only view returns into one lowering."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _embedding_rmsnorm_alias_kernel(
    embedding_ptr,
    input_ids_ptr,
    weight_ptr,
    raw_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS

    token = tl.load(input_ids_ptr + rows, mask=row_mask, other=0)
    table_offsets = token[:, None] * HIDDEN + cols[None, :]
    out_offsets = rows[:, None] * HIDDEN + cols[None, :]

    raw = tl.load(embedding_ptr + table_offsets, mask=row_mask[:, None], other=0.0)
    tl.store(raw_out_ptr + out_offsets, raw, mask=row_mask[:, None])

    x = raw.to(tl.float32)
    square_sum = tl.sum(tl.where(row_mask[:, None], x * x, 0.0), axis=1)
    inv_rms = tl.rsqrt(square_sum * (1.0 / HIDDEN) + 1.0e-6)
    normalized_bf16 = (x * inv_rms[:, None]).to(tl.bfloat16)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    out = (normalized_bf16.to(tl.float32) * weight[None, :]).to(tl.bfloat16)
    tl.store(norm_out_ptr + out_offsets, out, mask=row_mask[:, None])


@oracle_impl(hardware="B200", point="81ea203a", BLOCK_M=2, BLOCK_H=512, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="8d326be2", BLOCK_M=2, BLOCK_H=512, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    embedding_table, input_ids, weight, shape0, shape1, shape2 = inputs
    batch = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    rows = int(input_ids.numel())
    hidden = int(weight.numel())
    base_shape = (batch, seq, hidden)
    base_stride = (seq * hidden, hidden, 1)

    raw_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=embedding_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_rmsnorm_alias_kernel[(triton.cdiv(rows, BLOCK_M),)](
        embedding_table,
        input_ids,
        weight,
        raw_base,
        norm_base,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        raw_base,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
