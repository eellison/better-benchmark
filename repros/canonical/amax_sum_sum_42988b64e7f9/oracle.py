"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistillGPT2 shifted-label ignore-index cross-entropy scalar in Triton, including the constant-pad/slice/clone label shift semantics, bf16-to-fp32 loads from the captured padded row stride, natural-exp/log online row logsumexp, safe target gather, zero-filled ignored losses, f32 valid-count reduction, loss sum, and final f32 mean division, whereas Inductor lowers the decomposed pad/slice/view/amax/sub/exp/sum/log/gather/mask/count/div graph as generic reductions plus pointwise and gather kernels that materialize the full log-softmax intermediate; Inductor cannot do this today because its pattern library does not canonicalize shifted ignore-index language-model cross entropy with a scalar mean epilogue while preserving the strided bf16 logits layout and f32 arithmetic boundaries; the fix is NEW_PATTERN: add a guarded shifted cross-entropy lowering that fuses label shifting, online row logsumexp, masked gather/loss, valid-count reduction, and scalar mean codegen directly."""

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
    logits_ptr,
    labels_ptr,
    loss_ptr,
    valid_ptr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_COLS: tl.constexpr,
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


# c1ee6cd0: DistillGPT2 shifted labels i64[32,512], strided bf16 logits [32,512,50257].
@oracle_impl(hardware="B200", point="c1ee6cd0", BLOCK_N=4096, row_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, row_warps: int):
    labels, logits, shape_2d = inputs
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    seq_len = int(labels.shape[1])
    n_rows = int(labels.numel())

    logits_2d = logits.view(matrix_shape)
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
        logits_2d,
        labels,
        loss_per_row,
        valid_per_row,
        LOGITS_ROW_STRIDE=logits_2d.stride(0),
        SEQ_LEN=seq_len,
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
    return out
