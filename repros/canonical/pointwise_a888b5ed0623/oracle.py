"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XGLM int64 plus-two position shift, f32 row-index gather, and final `[1,128,1024]` contiguous view in one row-tiled Triton kernel that hoists each shifted row index across the 1024 hidden columns, whereas Inductor lowers the add/view/index/view chain through generic indirect-index scheduling; Inductor cannot do this today because its scheduler/codegen does not recognize constant-offset embedding gathers as a guarded row gather with per-token index hoisting and dense vectorized table loads; the fix is NEW_PATTERN: add an embedding-plus-constant-offset gather lowering that proves index bounds once per row and emits direct dense output stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ_LEN = 128
HIDDEN = 1024


@triton.jit
def _xglm_shifted_embedding_gather_kernel(
    ids_ptr,
    table_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    tokens = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    token_mask = tokens < seq_len
    col_mask = cols < hidden

    shifted = tl.load(ids_ptr + tokens, mask=token_mask, other=0) + 2
    values = tl.load(
        table_ptr + shifted[:, None] * hidden + cols[None, :],
        mask=token_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + tokens[:, None] * hidden + cols[None, :],
        values,
        mask=token_mask[:, None] & col_mask[None, :],
    )


# 56e670a4: XGLM f32 embedding table [2050,1024], int64 positions [1,128] shifted by +2.
@oracle_impl(hardware="B200", point="56e670a4", BLOCK_M=1, BLOCK_N=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    ids, table = inputs[:2]
    out = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=table.dtype,
    )
    _xglm_shifted_embedding_gather_kernel[(triton.cdiv(SEQ_LEN, BLOCK_M),)](
        ids,
        table,
        out,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
