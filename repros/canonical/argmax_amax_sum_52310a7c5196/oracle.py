"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-2 sequence-classification training tail in one Triton kernel, including the nonzero-token argmax scan, dynamic row-indexed bf16 pooled-logits side output, fp32 two-class stable log-softmax, explicit bf16 log-probability round followed by fp32 target gather, ignored-label valid-count and loss mean, label-valid mask, and one-hot label equality side output, whereas Inductor lowers the mask-derived argmax, advanced indexing, log-softmax reductions, gather, scalar reductions, and side-output masks through generic scheduling over materialized intermediates; Inductor cannot do this today because its pattern library does not canonicalize sequence-classification last-token gather feeding two-class cross-entropy while preserving the training side outputs and bf16 dtype boundary; the fix is NEW_PATTERN: add a sequence-classification cross-entropy lowering that fuses mask scan, dynamic logits gather, log-softmax, target loss reduction, and side-output stores into one planned kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _sequence_classification_tail_kernel(
    logits_ptr,
    input_ids_ptr,
    row_index_ptr,
    labels_ptr,
    argmax_out_ptr,
    pooled_out_ptr,
    count_out_ptr,
    loss_out_ptr,
    valid_mask_out_ptr,
    eq_out_ptr,
    BATCH: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batches = tl.arange(0, BLOCK_B)
    positions = tl.arange(0, BLOCK_N)
    batch_mask = batches < BATCH
    pos_mask = positions < SEQ

    token_offsets = batches[:, None] * SEQ + positions[None, :]
    tokens = tl.load(input_ids_ptr + token_offsets, mask=batch_mask[:, None] & pos_mask[None, :], other=0)
    candidate_positions = tl.where(tokens != 0, positions[None, :], 0)
    last_pos = tl.max(candidate_positions, axis=1)
    tl.store(argmax_out_ptr + batches, last_pos.to(tl.int64), mask=batch_mask)

    row_index = tl.load(row_index_ptr + batches, mask=batch_mask, other=0)
    logits_base = (row_index * SEQ + last_pos) * 2
    logit0_bf16 = tl.load(logits_ptr + logits_base, mask=batch_mask, other=0.0)
    logit1_bf16 = tl.load(logits_ptr + logits_base + 1, mask=batch_mask, other=0.0)
    tl.store(pooled_out_ptr + batches * 2, logit0_bf16, mask=batch_mask)
    tl.store(pooled_out_ptr + batches * 2 + 1, logit1_bf16, mask=batch_mask)

    logit0 = logit0_bf16.to(tl.float32)
    logit1 = logit1_bf16.to(tl.float32)
    row_max = tl.maximum(logit0, logit1)
    shifted0 = _f32_sub(logit0, row_max)
    shifted1 = _f32_sub(logit1, row_max)
    denom = _f32_add(libdevice.exp(shifted0), libdevice.exp(shifted1))
    log_denom = libdevice.log(denom)
    log_prob0 = _f32_sub(shifted0, log_denom).to(tl.bfloat16).to(tl.float32)
    log_prob1 = _f32_sub(shifted1, log_denom).to(tl.bfloat16).to(tl.float32)

    labels = tl.load(labels_ptr + batches, mask=batch_mask, other=-100)
    valid = labels != -100
    safe_labels = tl.where(valid, labels, 0)
    gathered = tl.where(safe_labels == 0, log_prob0, log_prob1)
    loss_terms = tl.where(valid & batch_mask, -gathered, 0.0)
    valid_count = tl.sum(tl.where(valid & batch_mask, 1.0, 0.0), axis=0)
    loss_sum = tl.sum(loss_terms, axis=0)
    tl.store(count_out_ptr, valid_count)
    tl.store(loss_out_ptr, _f32_div(loss_sum, valid_count))

    tl.store(valid_mask_out_ptr + batches, valid, mask=batch_mask)
    classes = tl.arange(0, 2)
    eq_values = safe_labels[:, None] == classes[None, :]
    tl.store(eq_out_ptr + batches[:, None] * 2 + classes[None, :], eq_values, mask=batch_mask[:, None])


# d21e6ec7: (T([8192,2], bf16), T([8,1024], i64), T([8], i64, gen=Index(8)), T([8], i64, gen=Index(2)), S([8,1024,2]), S([1,2]), S([8,2]))
@oracle_impl(hardware="B200", point="d21e6ec7", BLOCK_B=8, BLOCK_N=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_N: int, num_warps: int, num_stages: int):
    logits, input_ids, row_index, labels, _shape0, _shape1, _shape2 = inputs
    batch = int(labels.shape[0])
    seq = int(input_ids.shape[1])

    argmax_out = torch.empty_strided((batch,), (1,), device=logits.device, dtype=torch.int64)
    pooled = torch.empty_strided((batch, 2), (2, 1), device=logits.device, dtype=torch.bfloat16)
    count = torch.empty((), device=logits.device, dtype=torch.float32)
    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    valid_mask = torch.empty_strided((batch, 1), (1, 1), device=logits.device, dtype=torch.bool)
    eq = torch.empty_strided((batch, 2), (2, 1), device=logits.device, dtype=torch.bool)

    _sequence_classification_tail_kernel[(1,)](
        logits,
        input_ids,
        row_index,
        labels,
        argmax_out,
        pooled,
        count,
        loss,
        valid_mask,
        eq,
        BATCH=batch,
        SEQ=seq,
        BLOCK_B=BLOCK_B,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return argmax_out, pooled, count, loss, valid_mask, eq
