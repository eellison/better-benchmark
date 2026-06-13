"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete captured unsqueeze, int64 plus-two side output, and f32 embedding gather scope in one shape-specialized Triton kernel that writes both returned contiguous tensors directly, whereas Inductor lowers the shifted-position tensor and embedding through generic indirect-indexing code with per-element gather scheduling overhead; Inductor cannot do this today because it has no guarded constant-offset embedding-gather template that also preserves the observable shifted-index output; the fix is NEW_PATTERN: add an embedding-plus-constant-offset gather lowering that proves or guards index bounds once and emits direct dense gather stores plus the required index side output."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _embedding_add_side_output_kernel(
    ids_ptr,
    table_ptr,
    add_out_ptr,
    emb_out_ptr,
    HIDDEN: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    token_mask = tokens < SEQ_LEN
    col_mask = cols < HIDDEN

    shifted = tl.load(ids_ptr + tokens, mask=token_mask, other=0) + 2
    values = tl.load(
        table_ptr + shifted[:, None] * HIDDEN + cols[None, :],
        mask=token_mask[:, None] & col_mask[None, :],
        other=0.0,
    )

    tl.store(add_out_ptr + tokens, shifted, mask=token_mask)
    tl.store(
        emb_out_ptr + tokens[:, None] * HIDDEN + cols[None, :],
        values,
        mask=token_mask[:, None] & col_mask[None, :],
    )


# ca400fad: (T([1024], i64), T([1026,1024], f32))
@oracle_impl(hardware="B200", point="ca400fad", HIDDEN=1024, BLOCK_M=1, BLOCK_N=1024, num_warps=8, num_stages=3)
# df436a45: (T([1024], i64), T([1026,768], f32))
@oracle_impl(hardware="B200", point="df436a45", HIDDEN=768, BLOCK_M=1, BLOCK_N=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, HIDDEN, BLOCK_M, BLOCK_N, num_warps, num_stages):
    ids, table = inputs
    add_out = torch.empty_strided(
        (1, 1024),
        (1024, 1),
        device=ids.device,
        dtype=torch.int64,
    )
    emb_out = torch.empty_strided(
        (1, 1024, HIDDEN),
        (1024 * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=torch.float32,
    )

    _embedding_add_side_output_kernel[(triton.cdiv(1024, BLOCK_M),)](
        ids,
        table,
        add_out,
        emb_out,
        HIDDEN=HIDDEN,
        SEQ_LEN=1024,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, emb_out
