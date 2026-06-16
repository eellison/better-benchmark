"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J shifted-label causal-LM cross-entropy mean scope in a Triton online row-reduction plus scalar epilogue, including the constant-pad/slice label shift with ignore index -100, the returned bf16 `[1,128,50400]` logits view alias, fp32 stable row logsumexp over the 50400-column vocabulary, safe target gather, zero-loss ignore guard, valid-count reduction, loss reduction, and final fp32 scalar division, whereas Inductor lowers the pad/slice/ne/log-softmax/gather/masked-sum/count/div graph as generic reductions and pointwise/gather kernels that materialize the full log-softmax intermediate; Inductor cannot do this today because its online-softmax pattern matcher does not recognize shifted ignore-index cross entropy with a returned logits alias and scalar valid-count epilogue as one full-scope lowering; the fix is NEW_PATTERN: add a guarded shifted causal-LM cross-entropy template that fuses label shifting, online logsumexp, target gather, masked loss/count accumulation, and scalar mean codegen while preserving the returned view."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _shifted_xent_rows_kernel(
    labels_ptr,
    logits_ptr,
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

    loss = row_max + libdevice.log(denom) - target
    tl.store(loss_ptr + row, tl.where(is_valid, loss, 0.0))
    tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))


@triton.jit
def _mean_reduce_kernel(
    loss_ptr,
    valid_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_M)
    mask = offsets < N_ROWS
    losses = tl.load(loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    valid = tl.load(valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    total_loss = tl.sum(losses, axis=0)
    total_valid = tl.sum(valid, axis=0)
    tl.store(out_ptr, total_loss / total_valid)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# 627fbc0f: GPT-J shifted labels i64[1,128], bf16 logits [128,50400].
@oracle_impl(hardware="B200", point="627fbc0f", BLOCK_N=8192, row_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(logits.numel() // n_cols)
    seq_len = int(labels.shape[1])

    logits_view = logits.view(view_shape)
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
    out = torch.empty_strided((), (), device=logits.device, dtype=torch.float32)

    _shifted_xent_rows_kernel[(n_rows,)](
        labels,
        logits,
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
        out,
        N_ROWS=n_rows,
        BLOCK_M=triton.next_power_of_2(n_rows),
        num_warps=8,
        num_stages=3,
    )
    return logits_view, out
