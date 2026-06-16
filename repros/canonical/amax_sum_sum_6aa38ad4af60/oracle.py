"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete fp32 ignore-index cross-entropy mean scope from Repro.forward, including the label/logit views, stable row logsumexp over the 32000-column vocabulary, safe masked target gather, zero-loss ignore guard, valid-count reduction, loss reduction, final scalar division, and returned logits view alias, whereas Inductor lowers the view/ne/amax/sub/exp/sum/log/sub/where/gather/squeeze/neg/where/sum/count/div graph as generic row reductions, pointwise kernels, gather work, and scalar reductions that materialize the full log-softmax intermediate; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize ignore-index cross entropy mean with both gathered target loss and sibling valid-count reductions into one full-scope online row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add an Inductor lowering for log-softmax plus masked gather plus loss/count mean that emits an online cross-entropy row kernel and a small final reduction directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    loss_ptr,
    valid_ptr,
    ROW_STRIDE: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)
    row_start = row * ROW_STRIDE

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
    log_prob = (target - row_max) - log_denom
    loss = 0.0 - log_prob

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


# a49a430b: GoogleFnet inference labels i64[32,512], logits f32[16384,32000].
@oracle_impl(hardware="B200", point="a49a430b", BLOCK_N=8192, row_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(logits.numel() // n_cols)

    logits_view = logits.view(view_shape)
    labels_1d = labels.view(-1)
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

    _xent_rows_kernel[(n_rows,)](
        labels_1d,
        logits,
        loss_per_row,
        valid_per_row,
        ROW_STRIDE=logits.stride(0),
        N_COLS=n_cols,
        BLOCK_N=min(BLOCK_N, triton.next_power_of_2(n_cols)),
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
