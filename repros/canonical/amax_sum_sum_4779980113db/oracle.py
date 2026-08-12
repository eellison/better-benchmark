"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 ignore-index cross-entropy mean scope from Repro.forward, including the returned logits view alias, bf16-to-fp32 row amax/logsumexp over the full vocabulary, safe masked target gather after the observable bf16 log-softmax rounding boundary, bf16 loss summation, bf16 valid-count conversion, and final bf16 scalar division, whereas Inductor lowers the view/ne/amax/sub/exp/sum/log/sub/cast/gather/mask/count/div graph through generic row reductions, pointwise kernels, gather work, and scalar reductions that materialize a full log-softmax-sized intermediate; Inductor cannot do this today because its pattern library does not canonicalize ignore-index cross entropy mean with an aliasing logits output and bf16 scalar epilogue into one guarded row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add a guarded cross-entropy lowering that emits online logsumexp, masked target loss, valid count, and bf16 mean directly while returning the logits view as metadata."""

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
        denom = denom * libdevice.exp(_f32_sub(row_max, new_max))
        denom += tl.sum(libdevice.exp(_f32_sub(logits, new_max)), axis=0)
        row_max = new_max

    log_denom = libdevice.log(denom)
    logp = _f32_sub(_f32_sub(target, row_max), log_denom)
    loss = _f32_sub(0.0, logp.to(tl.bfloat16).to(tl.float32))
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


@oracle_impl(hardware="B200", point="6098747b", BLOCK_N=8192, num_warps=8)
@oracle_impl(hardware="B200", point="bba6ebb5", BLOCK_N=8192, num_warps=8)
@oracle_impl(hardware="B200", point="4f6a8ed5", BLOCK_N=8192, num_warps=8)
@oracle_impl(hardware="B200", point="29bccbec", BLOCK_N=8192, num_warps=8)
@oracle_impl(hardware="B200", point="f7a45f0f", BLOCK_N=8192, num_warps=8)
@oracle_impl(hardware="B200", point="8e79bb3c", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    n_rows = int(labels.numel())
    n_cols = int(shape_2d[1])

    logits_view = logits.view(view_shape)
    logits_2d = logits.view(n_rows, n_cols)
    labels_1d = labels.view(-1)
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
    div = torch.empty_strided((), (), device=logits.device, dtype=torch.bfloat16)

    _xent_rows_kernel[(n_rows,)](
        labels_1d,
        logits_2d,
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
        div,
        N_ROWS=n_rows,
        BLOCK_M=triton.next_power_of_2(n_rows),
        num_warps=8,
        num_stages=3,
    )
    return logits_view, div
