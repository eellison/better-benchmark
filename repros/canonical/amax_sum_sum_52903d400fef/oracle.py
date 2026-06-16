"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete shifted-label sliced-vocabulary Roberta causal-LM loss scope from Repro.forward, including the returned bf16 logits slice/view alias, returned int64 right-padded label tensor, f32 row amax/log-denominator side outputs, safe ignored-label gather, f32 valid-count conversion, and final f32 mean loss, whereas Inductor lowers the decomposed slice/view/pad/slice/clone/amax/sub/exp/sum/log/gather/mask/sum/count/div graph through generic reductions and pointwise/gather kernels that materialize the full log-softmax-sized intermediate; Inductor cannot do this today because its pattern library does not canonicalize shifted ignore-index language-model cross entropy with observable sibling pad/view/stat outputs while preserving the fp32 loss path and input-stride aliasing contract; the fix is NEW_PATTERN: add a guarded online cross-entropy lowering that recognizes the label-shift producers, sliced vocabulary, masked target gather, scalar valid-count epilogue, and returned logits/padded-label/stat side outputs."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _pad_right_kernel(
    labels_ptr,
    padded_ptr,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = BATCH * (SEQ_LEN + 1)
    mask = offsets < total
    col = offsets % (SEQ_LEN + 1)
    row = offsets // (SEQ_LEN + 1)
    in_bounds = mask & (col < SEQ_LEN)
    values = tl.load(labels_ptr + row * SEQ_LEN + col, mask=in_bounds, other=0)
    values = tl.where(col < SEQ_LEN, values, -100)
    tl.store(padded_ptr + offsets, values, mask=mask)


@triton.jit
def _shifted_xent_stats_rows_kernel(
    labels_ptr,
    logits_ptr,
    amax_ptr,
    log_ptr,
    loss_ptr,
    valid_ptr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    N_COLS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pos = row % SEQ_LEN
    has_next = pos < (SEQ_LEN - 1)

    label = tl.load(labels_ptr + row + 1, mask=has_next, other=-100)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)
    row_start = row * LOGITS_ROW_STRIDE

    target = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)

    row_max = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for block_start in tl.range(0, N_COLS, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        logits = tl.load(
            logits_ptr + row_start + cols,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * libdevice.exp(row_max - new_max)
        denom += tl.sum(libdevice.exp(logits - new_max), axis=0)
        row_max = new_max

    log_denom = libdevice.log(denom)
    loss = row_max + log_denom - target

    tl.store(amax_ptr + row, row_max)
    tl.store(log_ptr + row, log_denom)
    tl.store(loss_ptr + row, tl.where(is_valid, loss, 0.0))
    tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))


@triton.jit
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    count_out_ptr,
    div_out_ptr,
    N_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_M)
    mask = offsets < N_ROWS
    losses = tl.load(loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    valid = tl.load(valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    total_loss = tl.sum(losses, axis=0)
    total_valid = tl.sum(valid, axis=0)
    tl.store(count_out_ptr, total_valid)
    tl.store(div_out_ptr, total_loss / total_valid)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 68a9ca47: RobertaForCausalLM train bf16 logits [16384,50272] -> [:, :50265].
@oracle_impl(hardware="B200", point="68a9ca47", BLOCK_N=4096, row_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    n_rows = int(labels.numel())
    seq_len = int(labels.shape[1])
    batch = int(labels.shape[0])
    n_cols = int(shape_2d[1])

    logits_view = logits[:, :n_cols].view(view_shape)
    padded_labels = torch.empty_strided(
        (batch, seq_len + 1),
        (seq_len + 1, 1),
        device=labels.device,
        dtype=torch.int64,
    )
    amax = torch.empty_strided(
        (n_rows, 1),
        (1, 1),
        device=logits.device,
        dtype=torch.float32,
    )
    log = torch.empty_strided(
        (n_rows, 1),
        (1, 1),
        device=logits.device,
        dtype=torch.float32,
    )
    loss_per_row = torch.empty_strided(
        (n_rows,),
        (1,),
        device=logits.device,
        dtype=torch.float32,
    )
    valid_per_row = torch.empty_strided(
        (n_rows,),
        (1,),
        device=logits.device,
        dtype=torch.float32,
    )
    count = torch.empty_strided((), (), device=logits.device, dtype=torch.float32)
    div = torch.empty_strided((), (), device=logits.device, dtype=torch.float32)

    _pad_right_kernel[(triton.cdiv(batch * (seq_len + 1), 256),)](
        labels,
        padded_labels,
        BATCH=batch,
        SEQ_LEN=seq_len,
        BLOCK=256,
        num_warps=8,
        num_stages=3,
    )
    _shifted_xent_stats_rows_kernel[(n_rows,)](
        labels,
        logits,
        amax,
        log,
        loss_per_row,
        valid_per_row,
        LOGITS_ROW_STRIDE=logits.stride(0),
        N_COLS=n_cols,
        SEQ_LEN=seq_len,
        BLOCK_N=BLOCK_N,
        num_warps=row_warps,
        num_stages=3,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        count,
        div,
        N_ROWS=n_rows,
        BLOCK_M=triton.next_power_of_2(n_rows),
        num_warps=8,
        num_stages=3,
    )
    return logits_view, padded_labels, amax, log, count, div
