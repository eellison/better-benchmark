"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 token embedding gather, raw embedding side output, fp32 hidden-size RMS mean(square)+eps=1e-6 rsqrt, required bf16 normalization round, bf16 affine weight multiply, and three returned flattened alias views from one shape-specialized Triton row kernel, whereas Inductor lowers the embedding producer, generic mean reduction, RMSNorm epilogue, and sibling alias returns through separate general scheduling machinery; Inductor cannot do this today because norm-template canonicalization does not recognize an indexed token embedding feeding fixed-hidden RMSNorm while preserving the raw embedding side output and repeated view aliases from one normalized storage; the fix is NEW_PATTERN: add an embedding-RMSNorm alias template that folds indexed embedding loads, raw side-output stores, fp32 row RMS reduction, bf16 epilogue rounding, and alias-only view returns into one lowering."""

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
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    token = tl.load(input_ids_ptr + row)
    table_offsets = token * HIDDEN + cols
    out_offsets = row * HIDDEN + cols

    raw = tl.load(embedding_ptr + table_offsets)
    tl.store(raw_out_ptr + out_offsets, raw)

    x = raw.to(tl.float32)
    square_sum = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(square_sum / HIDDEN + 1.0e-6)
    normalized_bf16 = (x * inv_rms).to(tl.bfloat16)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    out = (normalized_bf16.to(tl.float32) * weight).to(tl.bfloat16)
    tl.store(norm_out_ptr + out_offsets, out)


# 79ba99e2: (T([151936,1024], bf16), T([1,1000], i64), T([1024], bf16), ...)
@oracle_impl(hardware="B200", point="79ba99e2", BLOCK_H=1024, num_warps=4, num_stages=2)
# c790717e: (T([32768,4096], bf16), T([1,1000], i64), T([4096], bf16), ...)
@oracle_impl(hardware="B200", point="c790717e", BLOCK_H=4096, num_warps=8, num_stages=2)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    embedding_table, input_ids, weight, shape0, shape1, shape2 = inputs
    rows = int(input_ids.numel())
    hidden = int(weight.numel())
    base_shape = (1, rows, hidden)
    base_stride = (rows * hidden, hidden, 1)

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

    _embedding_rmsnorm_alias_kernel[(rows,)](
        embedding_table,
        input_ids,
        weight,
        raw_base,
        norm_base,
        HIDDEN=hidden,
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
