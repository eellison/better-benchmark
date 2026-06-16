"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 GPT2 sequence-classification loss scope in one Triton kernel, including last nonzero-token argmax over the `[8,1024]` attention mask, the returned bf16 `[8,2]` gathered logits, f32 two-class logsumexp with natural libdevice exp/log, the required bf16 log-softmax rounding before safe target gather, ignore-index masking, bf16 loss summation, bf16 valid-count conversion, and final bf16 scalar division, whereas Inductor lowers the mask argmax, dynamic index gather, log-softmax, target gather, ignore mask, and scalar mean as separate generic reductions and pointwise kernels; Inductor cannot do this today because its scheduler does not canonicalize this sequence-classification cross-entropy tail with a visible gathered-logits side output and bf16 loss boundary into one full-scope plan; the fix is NEW_PATTERN: add a guarded last-token two-class cross-entropy lowering that emits the gathered logits and scalar loss directly while preserving ignore-index and bf16 rounding semantics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
SEQ_LEN = 1024
N_CLASSES = 2


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
    token_mask_ptr,
    gathered_ptr,
    loss_ptr,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batches = tl.arange(0, BLOCK_B)
    positions = tl.arange(0, BLOCK_S)
    token_offsets = batches[:, None] * BLOCK_S + positions[None, :]
    tokens = tl.load(token_mask_ptr + token_offsets)
    candidate_positions = tl.where(tokens != 0, positions[None, :], 0)
    last_positions = tl.max(candidate_positions, axis=1)

    row_offsets = (batches * BLOCK_S + last_positions) * 2
    logit0_bf16 = tl.load(logits_ptr + row_offsets).to(tl.bfloat16)
    logit1_bf16 = tl.load(logits_ptr + row_offsets + 1).to(tl.bfloat16)
    tl.store(gathered_ptr + batches * 2, logit0_bf16)
    tl.store(gathered_ptr + batches * 2 + 1, logit1_bf16)

    logit0 = logit0_bf16.to(tl.float32)
    logit1 = logit1_bf16.to(tl.float32)
    row_max = tl.maximum(logit0, logit1)
    shifted0 = _f32_sub(logit0, row_max)
    shifted1 = _f32_sub(logit1, row_max)
    exp0 = libdevice.exp(shifted0)
    exp1 = libdevice.exp(shifted1)
    denom = _f32_add(exp0, exp1)
    log_denom = libdevice.log(denom)
    logprob0 = _f32_sub(shifted0, log_denom).to(tl.bfloat16)
    logprob1 = _f32_sub(shifted1, log_denom).to(tl.bfloat16)

    labels = tl.load(labels_ptr + batches)
    valid = labels != -100
    safe_labels = tl.where(valid, labels, 0)
    selected = tl.where(safe_labels == 0, logprob0, logprob1)
    neg_loss = _f32_sub(0.0, selected.to(tl.float32)).to(tl.bfloat16)
    zero = tl.full((BLOCK_B,), 0.0, tl.float32).to(tl.bfloat16)
    losses = tl.where(valid, neg_loss, zero)

    loss_sum = tl.sum(losses.to(tl.float32), axis=0).to(tl.bfloat16)
    valid_count = tl.sum(tl.where(valid, 1.0, 0.0), axis=0).to(tl.bfloat16)
    mean_loss = _f32_div(loss_sum.to(tl.float32), valid_count.to(tl.float32)).to(
        tl.bfloat16
    )
    tl.store(loss_ptr, mean_loss)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 91cac6df: GPT2ForSequenceClassification infer, labels [8], bf16 logits [8192,2], mask [8,1024].
@oracle_impl(hardware="B200", point="91cac6df", BLOCK_B=8, BLOCK_S=1024, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_B: int,
    BLOCK_S: int,
    num_warps: int,
    num_stages: int,
):
    labels, logits, token_mask, _shape0 = inputs
    gathered = torch.empty_strided(
        (BATCH, N_CLASSES),
        _contiguous_stride((BATCH, N_CLASSES)),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    loss = torch.empty((), device=logits.device, dtype=torch.bfloat16)
    _last_token_xent_kernel[(1,)](
        labels,
        logits,
        token_mask,
        gathered,
        loss,
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return gathered, loss
