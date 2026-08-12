"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT bf16 `permute -> cat -> constant_pad_nd` scope by writing the fresh contiguous `[512,30528]` result directly from the transposed `[30522,128]` table, the `[384,30522]` table, and the six-column zero pad tail, whereas Inductor already lowers this isolated fixed row-cat plus right-pad graph to comparable final-output materialization work; Inductor cannot materially improve this case through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the exact contract is dominated by the required transpose/copy input traffic, dense bf16 output store, allocation, and pad zero-fill; the fix is BANDWIDTH_BOUND: keep this as an at-floor layout-copy measurement unless broader pointwise memory-codegen or launch/allocation changes move both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


TOP_ROWS = 128
BOTTOM_ROWS = 384
OUT_ROWS = TOP_ROWS + BOTTOM_ROWS
VOCAB = 30522
PADDED_VOCAB = 30528
OUT_SHAPE = (OUT_ROWS, PADDED_VOCAB)
OUT_STRIDE = (PADDED_VOCAB, 1)


@triton.jit
def _cat_pad_kernel(
    top_ptr,
    bottom_ptr,
    out_ptr,
    TOP_ROWS_: tl.constexpr,
    OUT_ROWS_: tl.constexpr,
    VOCAB_: tl.constexpr,
    PADDED_VOCAB_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    mask = (rows < OUT_ROWS_) & (cols < PADDED_VOCAB_)
    in_cat = cols < VOCAB_
    from_top = rows < TOP_ROWS_

    top_values = tl.load(
        top_ptr + cols * TOP_ROWS_ + rows,
        mask=mask & in_cat & from_top,
        other=0.0,
    )
    bottom_values = tl.load(
        bottom_ptr + (rows - TOP_ROWS_) * VOCAB_ + cols,
        mask=mask & in_cat & ~from_top,
        other=0.0,
    )
    values = tl.where(from_top, top_values, bottom_values)
    values = tl.where(in_cat, values, 0.0)
    tl.store(out_ptr + rows * PADDED_VOCAB_ + cols, values, mask=mask)


@oracle_impl(
    hardware="B200",
    point="0fc085ac",
    BLOCK_M=8,
    BLOCK_N=256,
    num_warps=8,
    num_stages=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    top, bottom, _pad = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=top.device,
        dtype=torch.bfloat16,
    )
    _cat_pad_kernel[
        (triton.cdiv(OUT_ROWS, BLOCK_M), triton.cdiv(PADDED_VOCAB, BLOCK_N))
    ](
        top,
        bottom,
        out,
        TOP_ROWS_=TOP_ROWS,
        OUT_ROWS_=OUT_ROWS,
        VOCAB_=VOCAB,
        PADDED_VOCAB_=PADDED_VOCAB,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
