"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete T5/MT5 bf16 dual RMSNorm/dropout backward and embedding-gradient update by copying the bf16 vocabulary-gradient base directly to the returned f32 table, computing each row producer once per branch, accumulating the two returned hidden-column reductions, and atomically adding only valid token-index contributors into the dense vocabulary output, whereas Inductor materializes two zero-filled `_unsafe_masked_index_put_accumulate` scatter intermediates and schedules the sibling reductions, dropout/RMSNorm pointwise producers, scatters, and dense adds as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured duplicate-index scatter-reduce node that shares rowwise RMSNorm reductions across hidden reductions and masked token-index accumulation into an existing dense base tensor; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that fuses row producer arithmetic, valid-index masking, hidden reductions, and direct dense base-update epilogues while preserving bf16 input and f32 output boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _init_outputs_kernel(
    base_ptr,
    out0_ptr,
    out1_ptr,
    out_vocab_ptr,
    TOTAL: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    values = tl.load(base_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    tl.store(out_vocab_ptr + offsets, values, mask=active)

    hidden_active = offsets < HIDDEN
    zeros = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out0_ptr + offsets, zeros, mask=hidden_active)
    tl.store(out1_ptr + offsets, zeros, mask=hidden_active)


@triton.jit
def _branch_scatter_reduce_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    weight_ptr,
    keep_ptr,
    saved_ptr,
    rstd_ptr,
    prior_ptr,
    index_ptr,
    out_reduce_ptr,
    out_vocab_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    VOCAB: tl.constexpr,
    DROPOUT_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    active = (row < ROWS) & (h < HIDDEN)
    offsets = row * HIDDEN + h

    added = (
        tl.load(mm0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )
    weight = tl.load(weight_ptr + h, mask=h < HIDDEN, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=active, other=0).to(tl.float32)
    saved = tl.load(saved_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row, mask=row < ROWS, other=0.0).to(tl.float32)
    prior = tl.load(prior_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    kept_saved = keep * saved * DROPOUT_SCALE
    dropped_rstd = kept_saved * rstd
    tl.atomic_add(
        out_reduce_ptr + h,
        added * dropped_rstd,
        sem="relaxed",
        mask=active,
    )

    weighted = added * weight
    row_dot = tl.sum(tl.where(active, weighted * kept_saved, 0.0), axis=0)
    prior_plus = prior + weighted * rstd
    rstd_cubed = rstd * rstd * rstd
    correction = ((row_dot * -0.5) * rstd_cubed / HIDDEN) * (kept_saved * 2.0)
    grad = (prior_plus + correction) * (keep * DROPOUT_SCALE)

    raw_index = tl.load(index_ptr + row, mask=row < ROWS, other=-1).to(tl.int64)
    valid_index = (raw_index >= 0) & (raw_index < VOCAB)
    tl.atomic_add(
        out_vocab_ptr + raw_index * HIDDEN + h,
        grad,
        sem="relaxed",
        mask=active & valid_index,
    )


# 4e691609: MT5 vocab=250112, rows=4096, hidden=512.
@oracle_impl(hardware="B200", point="4e691609", INIT_BLOCK=1024, BLOCK_H=512, num_warps_init=4, num_warps_branch=8)
# 0c10a950: T5 vocab=32128, rows=8192, hidden=512.
@oracle_impl(hardware="B200", point="0c10a950", INIT_BLOCK=1024, BLOCK_H=512, num_warps_init=4, num_warps_branch=8)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    num_warps_init: int,
    num_warps_branch: int,
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
        *_shape_params,
    ) = inputs

    vocab = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    rows = int(arg1_1.shape[0])
    total = vocab * hidden

    out0 = torch.empty((hidden,), device=arg0_1.device, dtype=torch.float32)
    out1 = torch.empty((hidden,), device=arg0_1.device, dtype=torch.float32)
    out_vocab = torch.empty((vocab, hidden), device=arg0_1.device, dtype=torch.float32)

    _init_outputs_kernel[(triton.cdiv(total, INIT_BLOCK),)](
        arg0_1,
        out0,
        out1,
        out_vocab,
        TOTAL=total,
        HIDDEN=hidden,
        BLOCK=INIT_BLOCK,
        num_warps=num_warps_init,
    )

    _branch_scatter_reduce_kernel[(rows,)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        out0,
        out_vocab,
        ROWS=rows,
        HIDDEN=hidden,
        VOCAB=vocab,
        DROPOUT_SCALE=1.1111111111111112,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_branch,
    )

    _branch_scatter_reduce_kernel[(rows,)](
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg6_1,
        arg15_1,
        arg16_1,
        arg9_1,
        out1,
        out_vocab,
        ROWS=rows,
        HIDDEN=hidden,
        VOCAB=vocab,
        DROPOUT_SCALE=1.1111111111111112,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps_branch,
    )

    return out0, out1, out_vocab
