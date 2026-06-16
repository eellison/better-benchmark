"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XLNet transposed-token embedding scope as one shape-specialized Triton gather, writing the single live bf16 `[512,16,1024]` embedding and returning the three flattened views plus the full slice as alias-only metadata, whereas Inductor lowers the transpose/clone, embedding gather, and view-only outputs through generic embedding and layout scheduling; Inductor cannot do this today because it lacks a guarded embedding-gather template that consumes the original `[batch,seq]` ids in transposed logical order while preserving the alias-only output scope without materializing the cloned id layout; the fix is NEW_PATTERN: add a shape-specialized embedding gather/layout template that folds the transpose/clone into the gather address calculation and registers dependent view outputs over the gathered activation."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 512
BATCH = 16
HIDDEN = 1024
ROWS = SEQ * BATCH


@triton.jit
def _xlnet_embedding_kernel(
    ids_ptr,
    table_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    n_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    seq = rows // batch_size
    batch = rows - seq * batch_size
    token = tl.load(ids_ptr + batch * seq_len + seq, mask=rows < n_rows, other=0)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < hidden_size)
    values = tl.load(
        table_ptr + token[:, None] * hidden_size + cols[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + rows[:, None] * hidden_size + cols[None, :], values, mask=mask)


def _run(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    ids, table, _shape0, _shape1, _shape2 = inputs
    embedding = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=table.dtype,
    )
    _xlnet_embedding_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        ids,
        table,
        embedding,
        seq_len=SEQ,
        batch_size=BATCH,
        hidden_size=HIDDEN,
        n_rows=ROWS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    flat0 = torch.as_strided(embedding, (ROWS, HIDDEN), (HIDDEN, 1))
    flat1 = torch.as_strided(embedding, (ROWS, HIDDEN), (HIDDEN, 1))
    flat2 = torch.as_strided(embedding, (ROWS, HIDDEN), (HIDDEN, 1))
    full_slice = torch.as_strided(
        embedding,
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
    )
    return embedding, flat0, flat1, flat2, full_slice


# hf/infer/XLNetLMHeadModel_5545c8e0: (T([16,512], i64, gen=Index(32000)), T([32000,1024], bf16), S([1,8192,1024]), S([1,8192,1024]), S([1,8192,1024]))
@oracle_impl(hardware="B200", point="5545c8e0", BLOCK_M=1, BLOCK_N=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    return _run(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
