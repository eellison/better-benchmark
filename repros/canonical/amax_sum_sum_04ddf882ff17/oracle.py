"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT biased sliced-vocabulary ignore-index cross-entropy scope in Triton, including the bf16 vocabulary-bias add with returned contiguous `[256,128,30522]` biased-logits view, returned bf16 `[32768,30522]` log-softmax tensor, safe masked target gather after the observable bf16 log-softmax rounding boundary, zero-filled ignored losses, f32 valid-count conversion, f32 loss sum, and final f32 mean division, whereas Inductor lowers the captured slice/view/add/cast/view/amax/sub/exp/sum/log/cast/gather/mask/count/div graph as generic pointwise, reduction, and gather kernels that materialize and reread full log-softmax-sized intermediates; Inductor cannot do this today because its pattern library does not canonicalize biased sliced-vocabulary ignore-index cross entropy with required biased-logits and log-softmax side outputs while preserving bf16 rounding boundaries and f32 scalar epilogue numerics; the fix is NEW_PATTERN: add a guarded biased log-softmax plus masked-gather lowering that stores both visible bf16 side outputs and emits online row logsumexp, masked loss/count, and scalar mean directly."""

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
def _biased_logsoftmax_xent_rows_kernel(
    logits_ptr,
    bias_ptr,
    labels_ptr,
    biased_out_ptr,
    logsoftmax_out_ptr,
    loss_ptr,
    valid_ptr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    input_row_start = row * LOGITS_ROW_STRIDE
    output_row_start = row * N_COLS
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    target_logit = tl.load(
        logits_ptr + input_row_start + safe_label,
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
    target = _f32_add(target_logit, target_bias).to(tl.bfloat16).to(tl.float32)

    row_max = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for block_start in tl.range(0, N_COLS, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        logits = tl.load(
            logits_ptr + input_row_start + cols,
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
        biased_bf16 = _f32_add(logits, bias).to(tl.bfloat16)
        tl.store(biased_out_ptr + output_row_start + cols, biased_bf16, mask=mask)
        biased = tl.where(mask, biased_bf16.to(tl.float32), -float("inf"))

        block_max = tl.max(biased, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = _f32_add(
            denom * libdevice.exp(_f32_sub(row_max, new_max)),
            tl.sum(libdevice.exp(_f32_sub(biased, new_max)), axis=0),
        )
        row_max = new_max

    log_denom = libdevice.log(denom)
    for block_start in tl.range(0, N_COLS, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        biased = tl.load(
            biased_out_ptr + output_row_start + cols,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        shifted = _f32_sub(biased, row_max)
        logp = _f32_sub(shifted, log_denom).to(tl.bfloat16)
        tl.store(logsoftmax_out_ptr + output_row_start + cols, logp, mask=mask)

    target_logp = _f32_sub(_f32_sub(target, row_max), log_denom).to(tl.bfloat16).to(tl.float32)
    loss = _f32_sub(0.0, target_logp)
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


# 8f164373: MobileBertForMaskedLM train, bf16 logits [32768,30528] -> [:, :30522] + f32 bias.
@oracle_impl(hardware="B200", point="8f164373", BLOCK_N=4096, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    logits, bias, labels, shape_3d, shape_2d, output_shape = inputs
    del output_shape
    biased_shape = _shape_tuple(shape_3d)
    matrix_shape = _shape_tuple(shape_2d)
    n_rows = int(matrix_shape[0])
    n_cols = int(matrix_shape[1])

    biased_logits = torch.empty_strided(
        biased_shape,
        _contiguous_stride(biased_shape),
        device=logits.device,
        dtype=torch.bfloat16,
    )
    logsoftmax = torch.empty_strided(
        matrix_shape,
        _contiguous_stride(matrix_shape),
        device=logits.device,
        dtype=torch.bfloat16,
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

    _biased_logsoftmax_xent_rows_kernel[(n_rows,)](
        logits,
        bias,
        labels,
        biased_logits,
        logsoftmax,
        loss_per_row,
        valid_per_row,
        LOGITS_ROW_STRIDE=logits.stride(0),
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
    return biased_logits, logsoftmax, count, div
