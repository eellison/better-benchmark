"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT bf16 masked-LM biased ignore-index cross-entropy scope from Repro.forward, including the logits slice that drops the final six columns, bf16 vocabulary-bias add with the returned contiguous `[256, 128, 30522]` biased-logits tensor, stable online row logsumexp over the bf16-rounded biased logits, safe masked target gather after the observable bf16 log-softmax cast, bf16 loss sum, bf16 valid-count conversion, and final bf16 scalar division, whereas Inductor lowers the decomposed view/slice/add/view/amax/sub/exp/sum/log/cast/gather/mask/sum/count/div graph through generic reductions and pointwise/gather kernels that materialize and reread full log-softmax-sized intermediates; Inductor cannot do this today because its pattern library does not canonicalize biased sliced-vocabulary ignore-index cross entropy with a required sibling logits output while preserving bf16 rounding boundaries; the fix is NEW_PATTERN: add a guarded biased log_softmax plus masked-gather mean lowering that emits an online row-reduction kernel with the side-output store and scalar epilogue directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _biased_xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    bias_ptr,
    logits_out_ptr,
    loss_ptr,
    valid_ptr,
    N_COLS: tl.constexpr,
    LOGITS_ROW_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * LOGITS_ROW_STRIDE
    out_row_start = row * N_COLS
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    target_logit = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)
    target_bias = tl.load(
        bias_ptr + safe_label,
        mask=is_valid,
        other=0.0,
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
        biased_bf16 = (logits + bias).to(tl.bfloat16)
        tl.store(logits_out_ptr + out_row_start + cols, biased_bf16, mask=mask)
        biased = tl.where(mask, biased_bf16.to(tl.float32), -float("inf"))

        block_max = tl.max(biased, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * libdevice.exp(row_max - new_max)
        denom += tl.sum(libdevice.exp(biased - new_max), axis=0)
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


# 6f08aca4: MobileBertForMaskedLM infer bf16 logits [32768,30528] -> [:, :30522] + bias.
@oracle_impl(hardware="B200", point="6f08aca4", BLOCK_N=4096, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    labels, logits, bias, shape_3d, shape_2d, output_shape = inputs
    del shape_2d
    n_rows = int(labels.numel())
    n_cols = int(bias.numel())
    out_shape = _shape_tuple(output_shape)

    logits_out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
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

    _biased_xent_rows_kernel[(n_rows,)](
        labels,
        logits,
        bias,
        logits_out,
        loss_per_row,
        valid_per_row,
        N_COLS=n_cols,
        LOGITS_ROW_STRIDE=logits.stride(0),
        BLOCK_N=BLOCK_N,
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
    return out, logits_out.view(_shape_tuple(shape_3d))
