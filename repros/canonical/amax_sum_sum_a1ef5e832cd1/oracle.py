"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeBERTaV2 bf16 sliced-vocabulary ignore-index cross-entropy mean returned by Repro.forward, including the flattened labels, logits slice that drops the final four columns, returned non-contiguous `[8,512,128100]` logits view, bf16-to-fp32 row logsumexp, safe masked target gather after the observable bf16 log-softmax cast, ignored-row zeroing, bf16 loss reduction, bf16 valid-count cast, and final bf16 scalar division in Triton kernels, whereas Inductor lowers the captured view/ne/slice/view/amax/sub/exp/sum/log/sub/cast/where/gather/neg/where/sum/count/div graph as generic reductions and pointwise/gather kernels that materialize the full bf16 log-softmax intermediate; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize sliced-vocabulary ignore-index cross entropy mean with a returned sliced logits view and sibling valid-count reduction while preserving the bf16 rounding boundaries; the fix is NEW_PATTERN: add an Inductor lowering for sliced bf16 log_softmax plus masked gather plus loss/count mean that emits an online row cross-entropy kernel and a small final reduction directly."""

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
    N_COLS: tl.constexpr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * LOGITS_ROW_STRIDE
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)
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

    logp = target - row_max - libdevice.log(denom)
    loss = 0.0 - logp.to(tl.bfloat16).to(tl.float32)
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

    total_loss = tl.sum(losses, axis=0).to(tl.bfloat16).to(tl.float32)
    total_valid = tl.sum(valid, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr, total_loss / total_valid)


def _view_shape(shape):
    return tuple(int(dim) for dim in shape)


# 9e036e55: DeBERTaV2 masked-LM bf16 logits [4096,128104] -> [:, :128100].
@oracle_impl(hardware="B200", point="9e036e55", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    labels, logits, shape_3d, shape_2d = inputs
    n_cols = int(shape_2d[1])
    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(_view_shape(shape_3d))
    logits_2d = logits_view.view(_view_shape(shape_2d))
    labels_1d = labels.view(-1)
    n_rows = int(logits_2d.shape[0])

    loss_per_row = torch.empty_strided(
        (n_rows,),
        (1,),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    valid_per_row = torch.empty_strided(
        (n_rows,),
        (1,),
        device=logits.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((), (), device=logits.device, dtype=torch.bfloat16)

    _xent_rows_kernel[(n_rows,)](
        labels_1d,
        logits_2d,
        loss_per_row,
        valid_per_row,
        N_COLS=n_cols,
        LOGITS_ROW_STRIDE=logits_2d.stride(0),
        BLOCK_N=min(BLOCK_N, triton.next_power_of_2(n_cols)),
        num_warps=num_warps,
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
