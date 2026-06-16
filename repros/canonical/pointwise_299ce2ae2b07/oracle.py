"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete M2M100 int64-to-int32, add-zero, int32 multiply, int64 add-one, advanced-index table gather, and final `[64,128,1024]` view as a row-tiled Triton gather that hoists each computed row index across the 1024 hidden columns. Inductor lowers the captured convert/add/mul/convert/add/view/index/view chain as a generic pointwise indirect-index kernel over all 8,388,608 output elements, which recomputes the same token-row arithmetic and reloads the same two integer inputs for every hidden column. Inductor cannot do this today because its scheduler/codegen does not recognize computed-index embedding gathers as structured row gathers with column-vectorized loads and hoisted index arithmetic. The fix is NEW_PATTERN: add a computed-index embedding-gather lowering that groups contiguous hidden columns by token row, applies negative-index wrapping/bounds semantics once per row, and emits direct dense table loads into the final layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS0 = 64
ROWS1 = 128
TOKENS = ROWS0 * ROWS1
VOCAB = 1026
HIDDEN = 1024


@triton.jit
def _computed_embedding_gather_kernel(
    cumsum_ptr,
    mask_ptr,
    table_ptr,
    out_ptr,
    n_tokens: tl.constexpr,
    vocab: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    token_mask = token_offsets < n_tokens
    col_mask = col_offsets < hidden

    cumsum_i32 = tl.load(cumsum_ptr + token_offsets, mask=token_mask, other=0).to(tl.int32)
    mask_i32 = tl.load(mask_ptr + token_offsets, mask=token_mask, other=0)
    row = (cumsum_i32 * mask_i32).to(tl.int64) + 1
    wrapped_row = tl.where(row < 0, row + vocab, row)
    tl.device_assert(
        (0 <= wrapped_row) & (wrapped_row < vocab),
        "index out of bounds: 0 <= wrapped_row < vocab",
    )

    load_offsets = wrapped_row[:, None] * hidden + col_offsets[None, :]
    store_offsets = token_offsets[:, None] * hidden + col_offsets[None, :]
    tile_mask = token_mask[:, None] & col_mask[None, :]
    values = tl.load(table_ptr + load_offsets, mask=tile_mask, other=0.0)
    tl.store(out_ptr + store_offsets, values, mask=tile_mask)


# (T([64,128], i64), T([64,128], i32), T([1026,1024], f32), S([64,128,1024]))
@oracle_impl(hardware="B200", point="01e90f6e", BLOCK_M=1, BLOCK_N=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    cumsum, mask, table, _shape = inputs
    out = torch.empty_strided(
        (ROWS0, ROWS1, HIDDEN),
        (ROWS1 * HIDDEN, HIDDEN, 1),
        device=table.device,
        dtype=table.dtype,
    )
    _computed_embedding_gather_kernel[(triton.cdiv(TOKENS, BLOCK_M),)](
        cumsum,
        mask,
        table,
        out,
        n_tokens=TOKENS,
        vocab=VOCAB,
        hidden=HIDDEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
