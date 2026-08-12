"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete LayoutLM dropout/layernorm-backward embedding-gradient tuple by copying the bf16 vocabulary base directly into the returned f32 vocabulary table, recomputing the shared rowwise producer with the captured bf16-to-f32 casts, accumulating both returned hidden-column reductions, and atomically adding the producer into the token-type, four spatial embedding, learned-position, and valid vocabulary scatter outputs with the captured masks and strided index inputs, whereas Inductor materializes the rowwise producer and lowers the sibling reductions, repeated `_unsafe_masked_index_put_accumulate` buffers, pairwise scatter adds, and dense base add as separate generic scheduled kernels; Inductor cannot do this today because scheduler/codegen has no multi-destination duplicate-index scatter-reduce node that shares row-local layernorm reductions across several masked embedding-gradient accumulators and a dense base-update epilogue; the fix is SCATTER_REDUCE: add a LayoutLM embedding-backward scatter-reduce lowering that fuses row algebra, validity masks, hidden reductions, multi-table indexed accumulation, batch-position accumulation, and direct dense vocabulary update while preserving bf16-to-f32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
SEGMENT_ROWS = 2
POSITION_ROWS = 512
TABLE_ROWS = 1024
VOCAB_ROWS = 30522
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _init_outputs_kernel(
    base_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_segment_ptr,
    out_table_a_ptr,
    out_table_b_ptr,
    out_table_c_ptr,
    out_table_d_ptr,
    out_position_ptr,
    out_vocab_ptr,
    VOCAB_TOTAL: tl.constexpr,
    TABLE_TOTAL: tl.constexpr,
    POSITION_TOTAL: tl.constexpr,
    SEGMENT_TOTAL: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)

    vocab_active = offsets < VOCAB_TOTAL
    base = tl.load(base_ptr + offsets, mask=vocab_active, other=0.0).to(tl.float32)
    tl.store(out_vocab_ptr + offsets, base, mask=vocab_active)

    table_active = offsets < TABLE_TOTAL
    tl.store(out_table_a_ptr + offsets, zeros, mask=table_active)
    tl.store(out_table_b_ptr + offsets, zeros, mask=table_active)
    tl.store(out_table_c_ptr + offsets, zeros, mask=table_active)
    tl.store(out_table_d_ptr + offsets, zeros, mask=table_active)

    position_active = offsets < POSITION_TOTAL
    segment_active = offsets < SEGMENT_TOTAL
    hidden_active = offsets < HIDDEN_
    tl.store(out_position_ptr + offsets, zeros, mask=position_active)
    tl.store(out_segment_ptr + offsets, zeros, mask=segment_active)
    tl.store(out_sum3_ptr + offsets, zeros, mask=hidden_active)
    tl.store(out_sum4_ptr + offsets, zeros, mask=hidden_active)


@triton.jit
def _row_scatter_reduce_kernel(
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    keep_ptr,
    weight_ptr,
    saved_ptr,
    row_scale_ptr,
    segment_index_ptr,
    table_a_index_ptr,
    table_b_index_ptr,
    table_c0_index_ptr,
    table_d0_index_ptr,
    table_c1_index_ptr,
    table_d1_index_ptr,
    position_index_ptr,
    vocab_index_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_segment_ptr,
    out_table_a_ptr,
    out_table_b_ptr,
    out_table_c_ptr,
    out_table_d_ptr,
    out_position_ptr,
    out_vocab_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    TABLE_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    ALIAS_S0: tl.constexpr,
    ALIAS_S1: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    active = h < HIDDEN_
    offsets = row * HIDDEN_ + h
    batch = row // SEQ_
    seq = row - batch * SEQ_

    source = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    source += tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    keep = tl.load(keep_ptr + offsets, mask=active, other=0).to(tl.float32)
    mul1 = source * (keep * DROPOUT_SCALE_)
    weight = tl.load(weight_ptr + h, mask=active, other=0.0).to(tl.float32)
    mul2 = mul1 * weight
    saved = tl.load(saved_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    row_sum = tl.sum(tl.where(active, mul2, 0.0), axis=0)
    row_dot = tl.sum(tl.where(active, mul2 * saved, 0.0), axis=0)
    row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
    producer = row_scale * ((mul2 * HIDDEN_) - row_sum - (saved * row_dot))

    tl.atomic_add(out_sum3_ptr + h, mul1 * saved, sem="relaxed", mask=active)
    tl.atomic_add(out_sum4_ptr + h, mul1, sem="relaxed", mask=active)

    segment_raw = tl.load(segment_index_ptr + row).to(tl.int64)
    segment_idx = tl.minimum(tl.maximum(segment_raw, 0), SEGMENT_ROWS_ - 1)
    tl.atomic_add(
        out_segment_ptr + segment_idx * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active,
    )

    table_a_raw = tl.load(table_a_index_ptr + row).to(tl.int64)
    table_a_valid = (table_a_raw >= 0) & (table_a_raw < TABLE_ROWS_) & (table_a_raw != -1)
    tl.atomic_add(
        out_table_a_ptr + table_a_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & table_a_valid,
    )

    table_b_raw = tl.load(table_b_index_ptr + row).to(tl.int64)
    table_b_valid = (table_b_raw >= 0) & (table_b_raw < TABLE_ROWS_) & (table_b_raw != -1)
    tl.atomic_add(
        out_table_b_ptr + table_b_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & table_b_valid,
    )

    alias_offset = batch * ALIAS_S0 + seq * ALIAS_S1
    table_c0_raw = tl.load(table_c0_index_ptr + alias_offset).to(tl.int64)
    table_c0_valid = (table_c0_raw >= 0) & (table_c0_raw < TABLE_ROWS_) & (table_c0_raw != -1)
    tl.atomic_add(
        out_table_c_ptr + table_c0_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & table_c0_valid,
    )

    table_c1_raw = tl.load(table_c1_index_ptr + alias_offset).to(tl.int64)
    table_c1_valid = (table_c1_raw >= 0) & (table_c1_raw < TABLE_ROWS_) & (table_c1_raw != -1)
    tl.atomic_add(
        out_table_c_ptr + table_c1_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & table_c1_valid,
    )

    table_d0_raw = tl.load(table_d0_index_ptr + alias_offset).to(tl.int64)
    table_d0_valid = (table_d0_raw >= 0) & (table_d0_raw < TABLE_ROWS_) & (table_d0_raw != -1)
    tl.atomic_add(
        out_table_d_ptr + table_d0_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & table_d0_valid,
    )

    table_d1_raw = tl.load(table_d1_index_ptr + alias_offset).to(tl.int64)
    table_d1_valid = (table_d1_raw >= 0) & (table_d1_raw < TABLE_ROWS_) & (table_d1_raw != -1)
    tl.atomic_add(
        out_table_d_ptr + table_d1_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & table_d1_valid,
    )

    position_raw = tl.load(position_index_ptr + seq).to(tl.int64)
    position_valid = (position_raw >= 0) & (position_raw < POSITION_ROWS_) & (position_raw != -1)
    tl.atomic_add(
        out_position_ptr + position_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & position_valid,
    )

    vocab_raw = tl.load(vocab_index_ptr + row).to(tl.int64)
    vocab_valid = (vocab_raw >= 0) & (vocab_raw < VOCAB_ROWS_) & (vocab_raw != 0)
    tl.atomic_add(
        out_vocab_ptr + vocab_raw * HIDDEN_ + h,
        producer,
        sem="relaxed",
        mask=active & vocab_valid,
    )


@oracle_impl(hardware="B200", point="1bcea4a2", INIT_BLOCK=1024, BLOCK_H=1024, num_warps_init=4, num_warps_row=8)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    num_warps_init: int,
    num_warps_row: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    out_sum3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_a = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_b = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_c = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_d = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    vocab_total = VOCAB_ROWS * HIDDEN
    table_total = TABLE_ROWS * HIDDEN
    position_total = POSITION_ROWS * HIDDEN
    segment_total = SEGMENT_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(vocab_total, INIT_BLOCK),)](
        arg0_1,
        out_sum3,
        out_sum4,
        out_segment,
        out_table_a,
        out_table_b,
        out_table_c,
        out_table_d,
        out_position,
        out_vocab,
        VOCAB_TOTAL=vocab_total,
        TABLE_TOTAL=table_total,
        POSITION_TOTAL=position_total,
        SEGMENT_TOTAL=segment_total,
        HIDDEN_=HIDDEN,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _row_scatter_reduce_kernel[(ROWS,)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        out_sum3,
        out_sum4,
        out_segment,
        out_table_a,
        out_table_b,
        out_table_c,
        out_table_d,
        out_position,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        POSITION_ROWS_=POSITION_ROWS,
        TABLE_ROWS_=TABLE_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ALIAS_S0=arg12_1.stride(0),
        ALIAS_S1=arg12_1.stride(1),
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_row,
    )

    return (
        out_sum3,
        out_sum4,
        out_segment,
        out_table_a,
        out_table_b,
        out_table_c,
        out_table_d,
        out_position,
        out_vocab,
    )
