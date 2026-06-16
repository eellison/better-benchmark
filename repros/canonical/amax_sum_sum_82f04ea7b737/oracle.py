"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Blenderbot biased-logits ignore-index cross-entropy scope in Triton, including the returned fp32 `[16,128,8008]` logits-plus-bias tensor, fp32 row amax and log-sum-exp side outputs, safe masked target gather, fp32 valid-count conversion, scalar loss sum, and final fp32 mean division, whereas Inductor lowers the bf16-to-fp32 add, amax/sub/exp/sum/log, gather, masking, and scalar reductions as generic scheduled regions that materialize and reread a full log-softmax-sized intermediate; Inductor cannot fuse this full returned-output envelope today because its pattern library does not canonicalize biased ignore-index cross entropy with visible biased logits and row-statistic side outputs as one guarded online-softmax schedule; the fix is NEW_PATTERN: add a biased log_softmax-plus-masked-gather lowering that stores the required logits side output while emitting row logsumexp, masked loss, valid count, and scalar mean directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _biased_xent_rows_kernel(
    logits_ptr,
    bias_ptr,
    labels_ptr,
    add_ptr,
    amax_ptr,
    log_ptr,
    loss_ptr,
    valid_ptr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * N_COLS
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    target_logit = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    target_bias = tl.load(
        bias_ptr + safe_label,
        mask=is_valid,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    target = target_logit + target_bias

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
        bias = tl.load(
            bias_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        add = logits + bias
        tl.store(add_ptr + row_start + cols, add, mask=mask)

        block_max = tl.max(add, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * libdevice.exp(row_max - new_max)
        denom += tl.sum(libdevice.exp(add - new_max), axis=0)
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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# a0e86354: Blenderbot train, bf16 logits [2048,8008] plus f32 vocab bias.
@oracle_impl(hardware="B200", point="a0e86354", BLOCK_N=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    logits, bias, labels, shape0, shape1 = inputs
    add_shape = _shape_tuple(shape0)
    matrix_shape = _shape_tuple(shape1)
    n_rows = int(labels.numel())
    n_cols = int(matrix_shape[1])

    add = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=logits.device,
        dtype=torch.float32,
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

    _biased_xent_rows_kernel[(n_rows,)](
        logits,
        bias,
        labels,
        add,
        amax,
        log,
        loss_per_row,
        valid_per_row,
        N_COLS=n_cols,
        BLOCK_N=min(BLOCK_N, triton.next_power_of_2(n_cols)),
        num_warps=num_warps,
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
    return add, amax, log, count, div
