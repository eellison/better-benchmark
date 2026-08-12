"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo sequence-classification last-token cross-entropy scope in one Triton kernel, including the nonzero-token argmax scan, selected bf16 logits side output, fp32 two-class stable log-softmax, explicit bf16 log-probability round before target gather, ignore-index masking, bf16 loss sum/count division, and returned `(pooled_logits, loss)` tuple, whereas Inductor lowers the mask-derived argmax, advanced indexing, log-softmax reductions, gather, mask, scalar sums, and tuple side output through generic scheduling over materialized intermediates; Inductor cannot do this today because its pattern library does not canonicalize sequence-classification last-token gather feeding two-class log-softmax plus ignored-label mean while preserving the selected-logits side output and bf16 dtype boundaries; the fix is NEW_PATTERN: add a sequence-classification cross-entropy lowering that fuses mask scan, logits gather, online log-softmax, target loss reduction, and side-output stores into one planned kernel."""

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
def _last_token_xent_kernel(
    labels_ptr,
    logits_ptr,
    input_ids_ptr,
    selected_ptr,
    loss_ptr,
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
    candidates = tl.where(tokens != 0, positions[None, :], 0)
    last_pos = tl.max(candidates, axis=1)

    row_offsets = (batches * SEQ + last_pos) * 2
    logit0_bf16 = tl.load(logits_ptr + row_offsets, mask=batch_mask, other=0.0)
    logit1_bf16 = tl.load(logits_ptr + row_offsets + 1, mask=batch_mask, other=0.0)
    tl.store(selected_ptr + batches * 2, logit0_bf16, mask=batch_mask)
    tl.store(selected_ptr + batches * 2 + 1, logit1_bf16, mask=batch_mask)

    logit0 = logit0_bf16.to(tl.float32)
    logit1 = logit1_bf16.to(tl.float32)
    row_max = tl.maximum(logit0, logit1)
    shifted0 = _f32_sub(logit0, row_max)
    shifted1 = _f32_sub(logit1, row_max)
    denom = _f32_add(libdevice.exp(shifted0), libdevice.exp(shifted1))
    log_denom = libdevice.log(denom)
    log_prob0 = _f32_sub(shifted0, log_denom).to(tl.bfloat16)
    log_prob1 = _f32_sub(shifted1, log_denom).to(tl.bfloat16)

    labels = tl.load(labels_ptr + batches, mask=batch_mask, other=-100)
    valid = labels != -100
    safe_labels = tl.where(valid, labels, 0)
    gathered = tl.where(safe_labels == 0, log_prob0, log_prob1)
    nll = (-gathered.to(tl.float32)).to(tl.bfloat16)
    loss_terms = tl.where(valid & batch_mask, nll, tl.full((BLOCK_B,), 0.0, tl.bfloat16))

    loss_sum_bf16 = tl.sum(loss_terms.to(tl.float32), axis=0).to(tl.bfloat16)
    valid_count = tl.sum(tl.where(valid & batch_mask, 1.0, 0.0), axis=0)
    valid_count_bf16 = valid_count.to(tl.bfloat16)
    loss = _f32_div(loss_sum_bf16.to(tl.float32), valid_count_bf16.to(tl.float32))
    tl.store(loss_ptr, loss.to(tl.bfloat16))


# eacfffdd: (T([32], i64, gen=Index(2)), T([4096,2], bf16), T([32,128], i64), S([32,128,2]))
@oracle_impl(hardware="B200", point="eacfffdd", BLOCK_B=32, BLOCK_N=128, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_N: int, num_warps: int, num_stages: int):
    labels, logits, input_ids, _shape_param_0 = inputs
    batch = int(labels.shape[0])
    seq = int(input_ids.shape[1])
    selected = torch.empty_strided(
        (batch, 2),
        (2, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    loss = torch.empty((), device=logits.device, dtype=torch.bfloat16)
    _last_token_xent_kernel[(1,)](
        labels,
        logits,
        input_ids,
        selected,
        loss,
        BATCH=batch,
        SEQ=seq,
        BLOCK_B=BLOCK_B,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return selected, loss
