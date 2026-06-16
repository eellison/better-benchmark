"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileBert masked-LM bf16 loss-backward fragment, including ignore-index label handling, scalar f32 division, the sparse one-hot producer's bf16 round trip, natural `exp` from the strided bf16 logits, the bf16 residual add, the returned f32 vocabulary sum, and the returned padded bf16 `[32768, 30528]` layout. Inductor currently materializes the expanded one-hot path and schedules the dense exp/add producer, column reduction, and padding output as separate generic regions around a very large dense tensor; it cannot do this today because scheduler/codegen does not form one full-scope plan that replaces the sparse label equality with a guarded row scalar while sharing the exact bf16 add producer with both observable outputs. The fix is COOPERATIVE_SPLIT_K: use a masked-LM backward template that materializes the required padded tensor once while accumulating per-column row-block partials for the sibling sum, preserving libdevice exp and bf16 cast boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 32768
VOCAB = 30522
PADDED_VOCAB = 30528


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _materialize_partials_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    residual_ptr,
    padded_out_ptr,
    partial_ptr,
    ROWS_N: tl.constexpr,
    VOCAB_N: tl.constexpr,
    PADDED_VOCAB_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS_N
    pad_col_mask = cols < PADDED_VOCAB_N
    vocab_col_mask = cols < VOCAB_N
    vocab_mask = row_mask[:, None] & vocab_col_mask[None, :]
    pad_mask = row_mask[:, None] & pad_col_mask[None, :]

    numerator = tl.load(numerator_ptr).to(tl.float32)
    denominator = tl.load(denominator_ptr).to(tl.float32)
    scale = _f32_div(numerator, denominator)

    label = tl.load(labels_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
    active_label = label != -100
    safe_label = tl.where(active_label, label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB_N)
    valid = active_label & in_vocab
    row_sum = tl.where(
        valid,
        _f32_sub(0.0, scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
            tl.float32
        ),
        0.0,
    )

    logits_offsets = rows[:, None] * PADDED_VOCAB_N + cols[None, :]
    dense_offsets = rows[:, None] * VOCAB_N + cols[None, :]
    logits = tl.load(logits_ptr + logits_offsets, mask=vocab_mask, other=0.0).to(
        tl.float32
    )
    exp_value = libdevice.exp(logits)

    sparse = tl.where(
        valid[:, None] & (cols[None, :] == safe_label[:, None]),
        row_sum[:, None],
        0.0,
    )
    dense_delta = _f32_sub(sparse, _f32_mul(exp_value, row_sum[:, None]))
    delta_bf16 = dense_delta.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + dense_offsets, mask=vocab_mask, other=0.0)
    out_value = _bf16_add(residual, delta_bf16)

    padded_offsets = rows[:, None] * PADDED_VOCAB_N + cols[None, :]
    zero_bf16 = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32).to(tl.bfloat16)
    padded_value = tl.where(vocab_col_mask[None, :], out_value, zero_bf16)
    tl.store(padded_out_ptr + padded_offsets, padded_value, mask=pad_mask)

    partial = tl.sum(tl.where(vocab_mask, out_value.to(tl.float32), 0.0), axis=0)
    partial_offsets = row_block * VOCAB_N + cols
    tl.store(partial_ptr + partial_offsets, partial, mask=vocab_col_mask)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_out_ptr,
    VOCAB_N: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_RB: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_RB)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < VOCAB_N)
    offsets = row_blocks[:, None] * VOCAB_N + cols[None, :]
    values = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    reduced = tl.sum(values, axis=0)
    tl.store(sum_out_ptr + cols, reduced, mask=cols < VOCAB_N)


# hf MobileBertForMaskedLM train, labels [256, 128], dense vocab [32768, 30522].
@oracle_impl(
    hardware="B200",
    point="57ba714d",
    BLOCK_R=64,
    BLOCK_C=32,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        *_shape_params,
    ) = inputs

    num_row_blocks = triton.cdiv(ROWS, BLOCK_R)
    device = arg3_1.device
    padded_out = torch.empty_strided(
        (ROWS, PADDED_VOCAB),
        (PADDED_VOCAB, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (num_row_blocks, VOCAB),
        (VOCAB, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((VOCAB,), (1,), device=device, dtype=torch.float32)

    _materialize_partials_kernel[
        (num_row_blocks, triton.cdiv(PADDED_VOCAB, BLOCK_C))
    ](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        padded_out,
        partial,
        ROWS_N=ROWS,
        VOCAB_N=VOCAB,
        PADDED_VOCAB_N=PADDED_VOCAB,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_sum_kernel[(triton.cdiv(VOCAB, BLOCK_C),)](
        partial,
        sum_out,
        VOCAB_N=VOCAB,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_RB=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )

    return sum_out, padded_out
