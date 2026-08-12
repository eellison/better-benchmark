"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 shifted-label causal-LM cross-entropy mean from Repro.forward, including the constant-pad/slice/clone label shift, returned logits `[batch, seq, vocab]` view alias, bf16-to-fp32 row logsumexp, f32 target gather, ignored-row zeroing, f32 valid-count reduction, f32 loss reduction, and final scalar mean division, whereas Inductor lowers the captured pad/slice/view/amax/sub/exp/sum/log/gather/mask/count/div graph as generic reductions plus pointwise/gather kernels that materialize and reread full log-softmax-sized intermediates; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize shifted-label ignore-index cross entropy with a visible logits view alias into one online row-reduction plus scalar epilogue while preserving the f32 gather boundary; the fix is NEW_PATTERN: add a guarded shifted-label cross-entropy lowering that emits online row loss/count accumulators directly and returns the alias-only logits view."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
    ROW_STRIDE: tl.constexpr,
    N_COLS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pos = row % SEQ_LEN
    label = tl.load(labels_ptr + row + 1, mask=pos < SEQ_LEN - 1, other=-100).to(tl.int64)
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
        denom = _f32_add(
            denom * libdevice.exp(_f32_sub(row_max, new_max)),
            tl.sum(libdevice.exp(_f32_sub(logits, new_max)), axis=0),
        )
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


def _launch(inputs, *, BLOCK_N: int, num_warps: int):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_cols = int(matrix_shape[1])
    n_rows = int(logits.numel() // n_cols)
    seq_len = int(labels.shape[1])

    logits_view = logits.view(view_shape)
    logits_2d = logits_view.view(n_rows, n_cols)
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
        ROW_STRIDE=logits_2d.stride(0),
        N_COLS=n_cols,
        SEQ_LEN=seq_len,
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


# 4137e900: MegatronBert causal-LM infer, rows=8192, vocab=29056.
@oracle_impl(hardware="B200", point="4137e900", BLOCK_N=4096, num_warps=8)
# 92fc17b6: XGLM causal-LM infer, rows=4096, vocab=256008.
@oracle_impl(hardware="B200", point="92fc17b6", BLOCK_N=8192, num_warps=16)
# ab0881e7: OPT causal-LM infer, rows=8192, vocab=50272.
@oracle_impl(hardware="B200", point="ab0881e7", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    return _launch(inputs, BLOCK_N=BLOCK_N, num_warps=num_warps)
