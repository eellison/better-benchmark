"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Blenderbot bf16 biased-logits ignore-index cross-entropy scope in Triton, including the returned bf16 `[16,128,8008]` logits-plus-bias tensor, bf16 add rounding before fp32 row logsumexp, natural-exp/log stable softmax, safe masked target gather after the observable bf16 log-softmax cast, bf16 ignored-loss zeroing, bf16 loss reduction, bf16 valid-count conversion, and final bf16 mean division, whereas Inductor lowers the captured view/add/amax/sub/exp/sum/log/cast/gather/mask/count/div graph as generic reductions and pointwise/gather kernels that materialize and reread the full bf16 log-softmax intermediate; Inductor cannot fuse this full returned-output envelope today because its scheduler/codegen pattern library does not canonicalize biased bf16 ignore-index cross entropy with visible biased logits and exact bf16 rounding boundaries as one guarded online-softmax schedule; the fix is NEW_PATTERN: add a biased bf16 log_softmax-plus-masked-gather lowering that stores the required logits side output while emitting row logsumexp, masked loss, valid count, and scalar mean directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _biased_bf16_xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    bias_ptr,
    add_ptr,
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
    target = (target_logit + target_bias).to(tl.bfloat16).to(tl.float32)

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
        add_bf16 = (logits + bias).to(tl.bfloat16)
        tl.store(add_ptr + row_start + cols, add_bf16, mask=mask)
        add = add_bf16.to(tl.float32)

        block_max = tl.max(add, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * libdevice.exp(row_max - new_max)
        denom += tl.sum(libdevice.exp(add - new_max), axis=0)
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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 05e1a070: Blenderbot infer, labels [16,128], bf16 logits [2048,8008], bf16 vocab bias [1,8008].
@oracle_impl(hardware="B200", point="05e1a070", BLOCK_N=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    labels, logits, bias, shape0, shape1 = inputs
    add_shape = _shape_tuple(shape0)
    matrix_shape = _shape_tuple(shape1)
    n_rows = int(labels.numel())
    n_cols = int(matrix_shape[1])

    add = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=logits.device,
        dtype=torch.bfloat16,
    )
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

    _biased_bf16_xent_rows_kernel[(n_rows,)](
        labels,
        logits,
        bias,
        add,
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
        out,
        N_ROWS=n_rows,
        BLOCK_M=triton.next_power_of_2(n_rows),
        num_warps=8,
        num_stages=3,
    )
    return add, out
