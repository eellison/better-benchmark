"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete OPT int64 token-id plus-two side output and f32 embedding gather scope in one shape-specialized Triton kernel, writing the returned contiguous `[4,2048]` shifted-index tensor and `[4,2048,768]` embedding tensor directly, whereas Inductor lowers the scalar index add and embedding through generic indirect-indexing code with separate observable side-output scheduling; Inductor cannot do this today because it has no guarded constant-offset embedding-gather template that preserves the shifted indices as a returned tensor while materializing the dense embedding output; the fix is NEW_PATTERN: add an embedding-plus-constant-offset lowering that proves or guards index bounds once and emits both the shifted-index side output and dense gather stores from one schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
SEQ_LEN = 2048
HIDDEN = 768
TOKENS = BATCH * SEQ_LEN


@triton.jit
def _embedding_add_side_output_kernel(
    ids_ptr,
    table_ptr,
    add_out_ptr,
    emb_out_ptr,
    hidden: tl.constexpr,
    tokens_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    token_mask = tokens < tokens_total
    col_mask = cols < hidden

    shifted = tl.load(ids_ptr + tokens, mask=token_mask, other=0) + 2
    values = tl.load(
        table_ptr + shifted[:, None] * hidden + cols[None, :],
        mask=token_mask[:, None] & col_mask[None, :],
        other=0.0,
    )

    tl.store(add_out_ptr + tokens, shifted, mask=token_mask)
    tl.store(
        emb_out_ptr + tokens[:, None] * hidden + cols[None, :],
        values,
        mask=token_mask[:, None] & col_mask[None, :],
    )


# (T([4,2048], i64), T([2050,768], f32))
@oracle_impl(hardware="B200", point="174c0900", BLOCK_M=1, BLOCK_N=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps, num_stages):
    ids, table = inputs
    add_out = torch.empty_strided(
        (BATCH, SEQ_LEN),
        (SEQ_LEN, 1),
        device=ids.device,
        dtype=torch.int64,
    )
    emb_out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=torch.float32,
    )

    _embedding_add_side_output_kernel[(triton.cdiv(TOKENS, BLOCK_M),)](
        ids,
        table,
        add_out,
        emb_out,
        hidden=HIDDEN,
        tokens_total=TOKENS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, emb_out
