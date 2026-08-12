"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete PLBart bf16 sliced-vocabulary ignore-index cross-entropy scope from Repro.forward, including the returned sliced `[16,1024,50005]` logits view with the original padded row stride, f32 row amax and natural-exp/log logsumexp side outputs, safe masked target gather after the observable bf16 log-softmax rounding boundary, ignored-row zero fill, f32 valid-count conversion, f32 loss sum, and final f32 mean division, whereas Inductor lowers the captured slice/view/amax/sub/exp/sum/log/sub/cast/gather/mask/sum/count/div graph as generic reductions and pointwise/gather kernels that materialize the full log-softmax-sized intermediate; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize sliced-vocabulary ignore-index cross entropy with required sibling row-statistic outputs, bf16 log-softmax rounding, and visible sliced-view aliasing; the fix is NEW_PATTERN: add a guarded bf16 log-softmax plus masked-gather lowering that emits an online row logsumexp/gather kernel and scalar reduction epilogue while preserving the returned view and f32 side outputs directly."""

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
def _xent_stats_rows_kernel(
    logits_ptr,
    labels_ptr,
    amax_ptr,
    log_ptr,
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
    sub = _f32_sub(target, row_max)
    logp_bf16 = _f32_sub(sub, log_denom).to(tl.bfloat16).to(tl.float32)
    loss = _f32_sub(0.0, logp_bf16)

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


# 353f43da: PLBart train, labels i64[16,1024], logits bf16[16384,50008] -> slice 50005.
@oracle_impl(hardware="B200", point="353f43da", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    logits, labels, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_rows = int(matrix_shape[0])
    n_cols = int(matrix_shape[1])

    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    logits_2d = logits_view.view(matrix_shape)
    labels_1d = labels.view(-1)
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

    _xent_stats_rows_kernel[(n_rows,)](
        logits_2d,
        labels_1d,
        amax,
        log,
        loss_per_row,
        valid_per_row,
        ROW_STRIDE=logits_2d.stride(0),
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
    return logits_view, amax, log, count, div
