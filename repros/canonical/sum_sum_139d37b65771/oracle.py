"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT masked-LM softmax-backward layout scope, including the captured per-row f32 sum over `arg0`, natural-exp update from `arg1`, visible bf16 rounding of the `[2048,20005]` intermediate, both zero-padded bf16 materializations (`[2048,20008]` and transposed `[20008,2048]`), and the returned vocab-column sum rounded through bf16 back to f32. Inductor currently lowers the row reduction, exp update, bf16 cast, two padding/layout materializations, and the sibling column reduction as separate generic regions over a large vocabulary tensor; it cannot do this today because scheduler/codegen does not keep the bf16 producer shared across both layout stores and the column reduction while preserving the explicit bf16 rounding and padded output scope. The fix is SCHEDULER_FUSION: add a guarded large-vocabulary backward/layout template that fuses the fixed row producer with its padded layout consumers and reduction epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 2048
VOCAB = 20005
PADDED_VOCAB = 20008
PAD = PADDED_VOCAB - VOCAB


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _materialize_partial_kernel(
    arg0_ptr,
    arg1_ptr,
    row_sum_ptr,
    padded_rows_ptr,
    padded_cols_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    VOCAB_: tl.constexpr,
    PADDED_VOCAB_: tl.constexpr,
    PAD_: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_
    col_mask = cols < VOCAB_
    mask = row_mask[:, None] & col_mask[None, :]

    offsets = rows[:, None] * VOCAB_ + cols[None, :]
    grad = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    log_probs = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    value = _f32_sub(grad, _f32_mul(libdevice.exp(log_probs), row_sum[:, None]))
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")
    value_f32 = tl.where(mask, value_bf16.to(tl.float32), 0.0)

    tl.store(
        padded_rows_ptr + rows[:, None] * PADDED_VOCAB_ + cols[None, :],
        value_bf16,
        mask=mask,
    )

    partial = tl.sum(value_f32, axis=0)
    tl.store(
        partial_ptr + cols * NUM_ROW_TILES + row_tile,
        partial,
        mask=col_mask,
    )
    tl.store(
        padded_cols_ptr + cols[None, :] * ROWS_ + rows[:, None],
        value_bf16,
        mask=mask,
    )
    pad = tl.arange(0, 4)
    pad_mask = pad < PAD_
    do_pad = col_tile == 0
    zero_rows = tl.full((BLOCK_M, 4), 0.0, tl.float32).to(tl.bfloat16)
    tl.store(
        padded_rows_ptr + rows[:, None] * PADDED_VOCAB_ + VOCAB_ + pad[None, :],
        zero_rows,
        mask=do_pad & row_mask[:, None] & pad_mask[None, :],
    )
    zero_cols = tl.full((4, BLOCK_M), 0.0, tl.float32).to(tl.bfloat16)
    tl.store(
        padded_cols_ptr + (VOCAB_ + pad[:, None]) * ROWS_ + rows[None, :],
        zero_cols,
        mask=do_pad & pad_mask[:, None] & row_mask[None, :],
    )


@triton.jit
def _finalize_column_sum_kernel(
    partial_ptr,
    out_sum_ptr,
    VOCAB_: tl.constexpr,
    NUM_ROW_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (cols[:, None] < VOCAB_) & (tiles[None, :] < NUM_ROW_TILES)
    values = tl.load(
        partial_ptr + cols[:, None] * NUM_ROW_TILES + tiles[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=1).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_sum_ptr + cols, sums.to(tl.float32), mask=cols < VOCAB_)


# BERT train/infer masked-LM vocab backward, [2048,20005] with 3-column padding.
@oracle_impl(
    hardware="B200",
    point="3746d560",
    BLOCK_M=64,
    BLOCK_N=128,
    FINAL_BLOCK_C=8,
    materialize_warps=8,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_C: int,
    materialize_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    device = arg0_1.device
    row_sum = torch.sum(arg0_1, dim=-1).reshape((ROWS,))
    padded_rows = torch.empty_strided(
        (ROWS, PADDED_VOCAB),
        (PADDED_VOCAB, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    padded_cols = torch.empty_strided(
        (PADDED_VOCAB, ROWS),
        (ROWS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    vocab_sum = torch.empty_strided((VOCAB,), (1,), device=device, dtype=torch.float32)

    num_row_tiles = triton.cdiv(ROWS, BLOCK_M)
    num_col_tiles = triton.cdiv(VOCAB, BLOCK_N)
    partial = torch.empty_strided(
        (VOCAB, num_row_tiles),
        (num_row_tiles, 1),
        device=device,
        dtype=torch.float32,
    )

    _materialize_partial_kernel[(num_row_tiles, num_col_tiles)](
        arg0_1,
        arg1_1,
        row_sum,
        padded_rows,
        padded_cols,
        partial,
        ROWS_=ROWS,
        VOCAB_=VOCAB,
        PADDED_VOCAB_=PADDED_VOCAB,
        PAD_=PAD,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=materialize_warps,
        num_stages=4,
    )
    _finalize_column_sum_kernel[(triton.cdiv(VOCAB, FINAL_BLOCK_C),)](
        partial,
        vocab_sum,
        VOCAB_=VOCAB,
        NUM_ROW_TILES=num_row_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        num_warps=final_warps,
        num_stages=3,
    )

    return padded_rows, padded_cols, vocab_sum
