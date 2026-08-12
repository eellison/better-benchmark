"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo sequence-classification tail in one Triton kernel, including reducing the attention mask to the last nonzero token per batch, gathering the visible bf16 two-class logits with the supplied batch index tensor, returning the argmax and gathered logits, evaluating the fp32 stable two-class log-softmax with the required bf16 round-trip before target gather, applying ignore-index masking with the captured scalar fill value, returning the fp32 valid-count and mean loss scalars, and emitting the `ne_2` and one-hot `eq` side masks, whereas Inductor lowers the argmax, dynamic gather, two-class log-softmax, masked NLL reductions, and side-mask stores through generic reduction, indexing, and pointwise schedules; Inductor cannot fuse this full returned-output envelope today because its pattern library does not canonicalize an argmax-selected sequence-classification cross-entropy tail with visible gathered logits and label masks as one guarded schedule; the fix is NEW_PATTERN: add a dedicated last-token two-class cross-entropy lowering that fuses mask argmax, logits gather, logsumexp/NLL mean, and side-output materialization while preserving bf16/fp32 cast boundaries."""

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
    logits_ptr,
    tokens_ptr,
    row_index_ptr,
    labels_ptr,
    ignored_value_ptr,
    argmax_ptr,
    selected_ptr,
    count_ptr,
    loss_ptr,
    ne2_ptr,
    eq_ptr,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_CLASSES: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batches = tl.arange(0, BLOCK_B)
    seq = tl.arange(0, BLOCK_S)
    batch_mask = batches < BATCH
    seq_mask = seq < SEQ_LEN

    token_offsets = batches[:, None] * SEQ_LEN + seq[None, :]
    tokens = tl.load(
        tokens_ptr + token_offsets,
        mask=batch_mask[:, None] & seq_mask[None, :],
        other=0,
        eviction_policy="evict_first",
    )
    positions = tl.where(tokens != 0, seq[None, :], 0)
    last_pos = tl.max(positions, axis=1)
    tl.store(argmax_ptr + batches, last_pos, mask=batch_mask)

    source_batch = tl.load(row_index_ptr + batches, mask=batch_mask, other=0)
    base = (source_batch * SEQ_LEN + last_pos) * N_CLASSES
    logit_bf16_0 = tl.load(
        logits_ptr + base,
        mask=batch_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    logit_bf16_1 = tl.load(
        logits_ptr + base + 1,
        mask=batch_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    tl.store(selected_ptr + batches * N_CLASSES, logit_bf16_0, mask=batch_mask)
    tl.store(selected_ptr + batches * N_CLASSES + 1, logit_bf16_1, mask=batch_mask)

    logit0 = logit_bf16_0.to(tl.float32)
    logit1 = logit_bf16_1.to(tl.float32)
    row_max = tl.maximum(logit0, logit1)
    sub0 = _f32_sub(logit0, row_max)
    sub1 = _f32_sub(logit1, row_max)
    denom = _f32_add(libdevice.exp(sub0), libdevice.exp(sub1))
    log_denom = libdevice.log(denom)
    logp0 = _f32_sub(sub0, log_denom).to(tl.bfloat16).to(tl.float32)
    logp1 = _f32_sub(sub1, log_denom).to(tl.bfloat16).to(tl.float32)

    label = tl.load(labels_ptr + batches, mask=batch_mask, other=-100)
    valid = label != -100
    safe_label = tl.where(valid, label, 0)
    target = tl.where(safe_label == 0, logp0, logp1)
    neg = 0.0 - target
    ignored_value = tl.load(ignored_value_ptr).to(tl.float32)
    row_loss = tl.where(valid, neg, ignored_value)
    row_loss = tl.where(batch_mask, row_loss, 0.0)
    valid_f32 = tl.where(batch_mask & valid, 1.0, 0.0)

    total_loss = tl.sum(row_loss, axis=0)
    total_valid = tl.sum(valid_f32, axis=0)
    tl.store(count_ptr, total_valid)
    tl.store(loss_ptr, _f32_div(total_loss, total_valid))

    tl.store(ne2_ptr + batches, valid, mask=batch_mask)
    tl.store(eq_ptr + batches * N_CLASSES, safe_label == 0, mask=batch_mask)
    tl.store(eq_ptr + batches * N_CLASSES + 1, safe_label == 1, mask=batch_mask)


@oracle_impl(
    hardware="B200",
    point="6bcef7c8",
    BLOCK_B=32,
    BLOCK_S=128,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_S: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    view_shape = tuple(int(dim) for dim in shape0)
    batch = int(view_shape[0])
    seq_len = int(view_shape[1])
    n_classes = int(view_shape[2])
    class_view_shape = tuple(int(dim) for dim in shape1)
    mask_shape = tuple(int(dim) for dim in shape2)

    argmax = torch.empty_strided(
        (batch,),
        (1,),
        device=arg0_1.device,
        dtype=torch.int64,
    )
    selected = torch.empty_strided(
        (batch, n_classes),
        (n_classes, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    count = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.float32)
    loss = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.float32)
    ne2 = torch.empty_strided(
        (batch, 1),
        (1, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    eq = torch.empty_strided(
        mask_shape,
        (n_classes, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    _last_token_xent_kernel[(1,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        argmax,
        selected,
        count,
        loss,
        ne2,
        eq,
        BATCH=batch,
        SEQ_LEN=seq_len,
        N_CLASSES=n_classes,
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=3,
    )
    return argmax, selected, count, loss, ne2, eq.view(mask_shape).view(mask_shape)
