"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Electra shifted-label sliced-vocabulary cross-entropy scope in Triton, including the returned bf16 `[64,512,30522]` sliced logits view alias with padded row stride, fp32 row logsumexp using natural exp/log over the first 30522 columns, safe shifted target gather with the last token ignored, zero-filled fp32 ignored losses, fp32 valid-count conversion, scalar loss sum, and final fp32 mean division, whereas Inductor lowers the captured pad/slice/clone/view/amax/sub/exp/sum/log/gather/mask/count/div graph as generic reductions and pointwise/gather kernels that materialize the full log-softmax intermediate; Inductor cannot fuse this full returned-output envelope today because its scheduler/codegen pattern library does not canonicalize shifted ignore-index language-model cross entropy with a visible sliced-logits alias and exact fp32 arithmetic boundaries as one guarded online-softmax schedule; the fix is NEW_PATTERN: add a guarded shifted causal-LM cross-entropy lowering that preserves the sliced logits view while emitting online row logsumexp, masked loss, valid count, and scalar mean directly."""

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
def _shifted_xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    loss_ptr,
    valid_ptr,
    LABEL_STRIDE_0: tl.constexpr,
    LABEL_STRIDE_1: tl.constexpr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // SEQ_LEN
    pos = row - batch * SEQ_LEN
    has_next = pos < (SEQ_LEN - 1)

    label = tl.load(
        labels_ptr + batch * LABEL_STRIDE_0 + (pos + 1) * LABEL_STRIDE_1,
        mask=has_next,
        other=-100,
    )
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


# a7052938: Electra infer shifted labels i64[64,512], bf16 padded logits [32768,30528].
@oracle_impl(hardware="B200", point="a7052938", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(labels.numel())
    seq_len = int(labels.shape[1])

    logits_slice = logits[:, :n_cols]
    logits_view = logits_slice.view(view_shape)
    logits_2d = logits_view.view(matrix_shape)
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
        logits_2d,
        loss_per_row,
        valid_per_row,
        LABEL_STRIDE_0=labels.stride(0),
        LABEL_STRIDE_1=labels.stride(1),
        LOGITS_ROW_STRIDE=logits_2d.stride(0),
        SEQ_LEN=seq_len,
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
    return logits_view, out
