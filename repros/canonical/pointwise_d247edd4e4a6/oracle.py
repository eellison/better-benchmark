"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete MobileBERT bf16/fp32 scatter-gather scope, including the returned f32 vocabulary-tail slice, the shifted-neighbor hidden producer over `[256,128,128]`, the captured validity mask on token ids, duplicate-preserving `_unsafe_masked_index_put_accumulate` into `[30522,128]`, and the final add of the transposed bf16 vocabulary table. Inductor lowers the slices, pad/scatter shifts, validity predicates, indexed accumulation, and dense add as generic pointwise/scatter regions; it cannot express this local pattern as one structured duplicate-index scatter-reduce with the required bf16-to-f32 cast boundaries and visible dense side output. The fix is SCATTER_REDUCE: add a MobileBERT embedding-gradient scatter lowering that initializes the dense destination directly, folds the shifted hidden producer into the masked token scatter, and adds the base table after the accumulated scatter."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 128
INPUT_HIDDEN = 384
ARG0_ROWS = 512
ARG0_COLS = 30528
VOCAB = 30522
TAIL_ROWS = 384
ROWS = BATCH * TOKENS
OUT0_NUMEL = TAIL_ROWS * VOCAB
OUT1_NUMEL = VOCAB * HIDDEN


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _init_and_tail_kernel(
    vocab_ptr,
    tail_out,
    scatter_out,
    n_tail: tl.constexpr,
    n_scatter: tl.constexpr,
    vocab_cols: tl.constexpr,
    vocab: tl.constexpr,
    hidden: tl.constexpr,
    tail_row_offset: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    scatter_mask = offsets < n_scatter
    tl.store(scatter_out + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=scatter_mask)

    tail_mask = offsets < n_tail
    tail_row = offsets // vocab
    tail_col = offsets - tail_row * vocab
    source = (tail_row + tail_row_offset) * vocab_cols + tail_col
    tail_value = tl.load(vocab_ptr + source, mask=tail_mask, other=0.0).to(tl.float32)
    tl.store(tail_out + offsets, tail_value, mask=tail_mask)


@triton.jit
def _scatter_kernel(
    hidden_ptr,
    token_id_ptr,
    scatter_out,
    rows_total: tl.constexpr,
    tokens: tl.constexpr,
    input_hidden: tl.constexpr,
    hidden: tl.constexpr,
    vocab: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < rows_total
    col_mask = cols < hidden
    mask = row_mask & col_mask

    token_pos = rows - (rows // tokens) * tokens
    token_id = tl.load(token_id_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    valid_token = (token_id >= 0) & (token_id < vocab) & (token_id != 0)

    mid = tl.load(
        hidden_ptr + rows * input_hidden + hidden + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    next_val = tl.load(
        hidden_ptr + (rows + 1) * input_hidden + 2 * hidden + cols,
        mask=mask & (token_pos < (tokens - 1)),
        other=0.0,
    ).to(tl.float32)
    prev_val = tl.load(
        hidden_ptr + (rows - 1) * input_hidden + cols,
        mask=mask & (token_pos > 0),
        other=0.0,
    ).to(tl.float32)

    value = _add_rn(_add_rn(mid, next_val), prev_val)
    tl.atomic_add(
        scatter_out + token_id * hidden + cols,
        value,
        sem="relaxed",
        mask=mask & valid_token,
    )


@triton.jit
def _add_base_kernel(
    vocab_ptr,
    scatter_out,
    n_scatter: tl.constexpr,
    vocab_cols: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_scatter
    vocab_col = offsets // hidden
    hidden_row = offsets - vocab_col * hidden
    base = tl.load(
        vocab_ptr + hidden_row * vocab_cols + vocab_col,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    accum = tl.load(scatter_out + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(scatter_out + offsets, _add_rn(base, accum), mask=mask)


def _launch(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps_init: int,
    num_warps_scatter: int,
    num_warps_base: int,
):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    device = arg0_1.device
    tail = torch.empty_strided(
        (TAIL_ROWS, VOCAB),
        (VOCAB, 1),
        device=device,
        dtype=torch.float32,
    )
    scatter = torch.empty_strided(
        (VOCAB, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )

    init_grid = (triton.cdiv(max(OUT0_NUMEL, OUT1_NUMEL), INIT_BLOCK),)
    _init_and_tail_kernel[init_grid](
        arg0_1,
        tail,
        scatter,
        n_tail=OUT0_NUMEL,
        n_scatter=OUT1_NUMEL,
        vocab_cols=ARG0_COLS,
        vocab=VOCAB,
        hidden=HIDDEN,
        tail_row_offset=ARG0_ROWS - TAIL_ROWS,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _scatter_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        arg1_1,
        arg2_1,
        scatter,
        rows_total=ROWS,
        tokens=TOKENS,
        input_hidden=INPUT_HIDDEN,
        hidden=HIDDEN,
        vocab=VOCAB,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_scatter,
    )

    _add_base_kernel[(triton.cdiv(OUT1_NUMEL, INIT_BLOCK),)](
        arg0_1,
        scatter,
        n_scatter=OUT1_NUMEL,
        vocab_cols=ARG0_COLS,
        hidden=HIDDEN,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_base,
    )
    return tail, scatter


@oracle_impl(
    hardware="B200",
    point="11f71568",
    INIT_BLOCK=1024,
    BLOCK_M=1,
    BLOCK_H=128,
    num_warps_init=4,
    num_warps_scatter=4,
    num_warps_base=4,
)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps_init: int,
    num_warps_scatter: int,
    num_warps_base: int,
):
    return _launch(
        inputs,
        INIT_BLOCK=INIT_BLOCK,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps_init=num_warps_init,
        num_warps_scatter=num_warps_scatter,
        num_warps_base=num_warps_base,
    )
