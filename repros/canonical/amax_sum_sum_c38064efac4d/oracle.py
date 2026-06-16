"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J shifted-label causal-LM cross-entropy scope in a Triton online row-reduction plus scalar epilogue, including the returned bf16 `[1,128,50400]` logits view alias, returned i64 constant-pad labels, fp32 row amax and log-sum-exp denominator side outputs, safe target gather, invalid-row fill from the captured f32 scalar, valid-count conversion, and final fp32 mean loss, whereas Inductor lowers the pad/slice/view/amax/sub/exp/sum/log/gather/masked-sum/count/div graph as generic reductions and pointwise/gather kernels that materialize the full log-softmax intermediate; Inductor cannot do this today because its online-softmax pattern matcher does not recognize shifted ignore-index cross entropy with returned row-statistic and padded-label side outputs as one full-scope lowering; the fix is NEW_PATTERN: add a guarded shifted causal-LM cross-entropy template that fuses label shifting, online logsumexp, target gather, masked loss/count accumulation, scalar mean codegen, and side-output stores while preserving view aliasing and fp32 numerics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _shifted_xent_stats_rows_kernel(
    logits_ptr,
    labels_ptr,
    fill_ptr,
    padded_ptr,
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

    current_label = tl.load(labels_ptr + row)
    tl.store(padded_ptr + row, current_label)
    if row == 0:
        tl.store(padded_ptr + SEQ_LEN, -100)

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
    fill = tl.load(fill_ptr).to(tl.float32)

    tl.store(amax_ptr + row, row_max)
    tl.store(log_ptr + row, log_denom)
    tl.store(loss_ptr + row, tl.where(is_valid, loss, fill))
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


# a71ab921: GPT-J train shifted labels i64[1,128], bf16 logits [128,50400].
@oracle_impl(hardware="B200", point="a71ab921", BLOCK_N=8192, row_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    arg0_1, arg1_1, arg2_1, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(arg0_1.numel() // n_cols)
    seq_len = int(arg1_1.shape[1])

    logits_view = arg0_1.view(view_shape)
    padded = torch.empty_strided(
        (int(arg1_1.shape[0]), seq_len + 1),
        (seq_len + 1, 1),
        device=arg1_1.device,
        dtype=torch.int64,
    )
    amax = torch.empty_strided(
        (n_rows, 1),
        (1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    log = torch.empty_strided(
        (n_rows, 1),
        (1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    loss_per_row = torch.empty_strided(
        (n_rows,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    valid_per_row = torch.empty_strided(
        (n_rows,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    count = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.float32)
    div = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.float32)

    _shifted_xent_stats_rows_kernel[(n_rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        padded,
        amax,
        log,
        loss_per_row,
        valid_per_row,
        LOGITS_ROW_STRIDE=arg0_1.stride(0),
        N_COLS=n_cols,
        SEQ_LEN=seq_len,
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
    return logits_view, padded, amax, log, count, div
