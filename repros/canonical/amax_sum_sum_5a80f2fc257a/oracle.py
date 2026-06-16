"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Electra shifted-label sliced-vocabulary cross-entropy scope in Triton, including the returned bf16 `[64,512,30522]` logits view alias, returned contiguous i64 padded labels, bf16-to-fp32 row amax and natural-exp/log logsumexp side outputs, safe ignore-index target gather, zero-filled ignored losses, f32 valid-count conversion, scalar loss sum, and final f32 mean division, whereas Inductor lowers the slice/view/pad/clone/amax/sub/exp/sum/log/gather/mask/count/div graph as generic reductions and pointwise/gather kernels that materialize the full log-softmax intermediate; Inductor cannot do this today because its pattern library does not canonicalize shifted ignore-index language-model cross entropy with observable padded-label and row-statistic side outputs while preserving the sliced logits stride, returned alias, and f32 arithmetic boundaries; the fix is NEW_PATTERN: add a guarded shifted causal-LM cross-entropy lowering that fuses label shifting, padded-label side-output stores, online row logsumexp, masked gather/loss, valid-count reduction, and scalar mean epilogue directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _shifted_xent_stats_kernel(
    logits_ptr,
    labels_ptr,
    padded_labels_ptr,
    amax_ptr,
    log_ptr,
    loss_ptr,
    valid_ptr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_COLS: tl.constexpr,
    PADDED_SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // SEQ_LEN
    pos = row - batch * SEQ_LEN
    has_next = pos < (SEQ_LEN - 1)

    original_label = tl.load(labels_ptr + row)
    tl.store(padded_labels_ptr + batch * PADDED_SEQ_LEN + pos, original_label)
    if pos == (SEQ_LEN - 1):
        tl.store(padded_labels_ptr + batch * PADDED_SEQ_LEN + SEQ_LEN, -100)

    label = tl.load(labels_ptr + row + 1, mask=has_next, other=-100)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)
    row_start = row * LOGITS_ROW_STRIDE

    target = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
        eviction_policy="evict_first",
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
    logp = _f32_sub(_f32_sub(target, row_max), log_denom)
    loss = _f32_sub(0.0, logp)

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


# a7052938: Electra train shifted labels i64[64,512], bf16 logits [32768,30528].
@oracle_impl(hardware="B200", point="a7052938", BLOCK_N=8192, row_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    batch = int(labels.shape[0])
    seq_len = int(labels.shape[1])
    n_rows = batch * seq_len

    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    logits_2d = logits_view.view(matrix_shape)
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

    _shifted_xent_stats_kernel[(n_rows,)](
        logits_2d,
        labels,
        padded_labels,
        amax,
        log,
        loss_per_row,
        valid_per_row,
        LOGITS_ROW_STRIDE=logits_2d.stride(0),
        SEQ_LEN=seq_len,
        N_COLS=n_cols,
        PADDED_SEQ_LEN=seq_len + 1,
        BLOCK_N=min(BLOCK_N, triton.next_power_of_2(n_cols)),
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
