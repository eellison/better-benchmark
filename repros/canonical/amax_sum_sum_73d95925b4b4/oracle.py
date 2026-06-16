"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J question-answering training loss scope in one Triton kernel, including the `[128,8]` input slice to the first two start/end columns, `[1,128,2]` view/split, returned contiguous bf16 start/end clones, clamped start/end labels, fp32 stable row amax/libdevice.exp/sum/log for each head, explicit bf16 log-softmax rounding before the f32 gather, returned f32 amax/log side tensors, returned bool valid masks, f32 ignored-label fallback scalar, f32 valid-count divisions, and final f32 average, whereas Inductor lowers the slice/split/clone/clamp/log-softmax/gather/masked-loss/count/div/average graph through generic pointwise, reduction, and copy kernels; Inductor cannot do this today because its pattern library does not canonicalize small two-head QA log_softmax plus clamped-label gather with visible clone and row-stat side outputs into one full-scope lowering; the fix is NEW_PATTERN: add a guarded QA cross-entropy template that fuses clone stores, online logsumexp, target gather, ignore-index handling, side-output stores, and scalar epilogue while preserving the returned-output envelope."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _qa_two_head_train_kernel(
    logits_ptr,
    start_label_ptr,
    end_label_ptr,
    ignored_scalar_ptr,
    start_out_ptr,
    end_out_ptr,
    start_amax_ptr,
    start_log_ptr,
    start_valid_ptr,
    end_amax_ptr,
    end_log_ptr,
    end_valid_ptr,
    loss_out_ptr,
    INPUT_STRIDE_ROW: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N_COLS

    start_bf16 = tl.load(
        logits_ptr + cols * INPUT_STRIDE_ROW,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    end_bf16 = tl.load(
        logits_ptr + cols * INPUT_STRIDE_ROW + 1,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(start_out_ptr + cols, start_bf16, mask=mask)
    tl.store(end_out_ptr + cols, end_bf16, mask=mask)

    start = tl.where(mask, start_bf16.to(tl.float32), -float("inf"))
    end = tl.where(mask, end_bf16.to(tl.float32), -float("inf"))

    start_max = tl.max(start, axis=0)
    end_max = tl.max(end, axis=0)
    start_shifted = start - start_max
    end_shifted = end - end_max
    start_log_denom = libdevice.log(tl.sum(libdevice.exp(start_shifted), axis=0))
    end_log_denom = libdevice.log(tl.sum(libdevice.exp(end_shifted), axis=0))
    start_logp_bf16 = (start_shifted - start_log_denom).to(tl.bfloat16)
    end_logp_bf16 = (end_shifted - end_log_denom).to(tl.bfloat16)

    raw_start_label = tl.load(start_label_ptr)
    raw_end_label = tl.load(end_label_ptr)
    start_label = tl.minimum(tl.maximum(raw_start_label, 0), N_COLS)
    end_label = tl.minimum(tl.maximum(raw_end_label, 0), N_COLS)
    valid_start = start_label != N_COLS
    valid_end = end_label != N_COLS
    safe_start_label = tl.where(valid_start, start_label, 0)
    safe_end_label = tl.where(valid_end, end_label, 0)

    gathered_start = tl.sum(
        tl.where(cols == safe_start_label, start_logp_bf16.to(tl.float32), 0.0),
        axis=0,
    )
    gathered_end = tl.sum(
        tl.where(cols == safe_end_label, end_logp_bf16.to(tl.float32), 0.0),
        axis=0,
    )

    ignored = tl.load(ignored_scalar_ptr).to(tl.float32)
    start_loss = tl.where(valid_start, 0.0 - gathered_start, ignored)
    end_loss = tl.where(valid_end, 0.0 - gathered_end, ignored)
    start_count = tl.where(valid_start, 1.0, 0.0)
    end_count = tl.where(valid_end, 1.0, 0.0)
    start_mean = start_loss / start_count
    end_mean = end_loss / end_count
    loss = (start_mean + end_mean) / 2.0

    tl.store(start_amax_ptr, start_max)
    tl.store(start_log_ptr, start_log_denom)
    tl.store(start_valid_ptr, valid_start)
    tl.store(end_amax_ptr, end_max)
    tl.store(end_log_ptr, end_log_denom)
    tl.store(end_valid_ptr, valid_end)
    tl.store(loss_out_ptr, loss)


# fe37f3bb: GPT-J QA train bf16 logits [128,8], start/end labels i64[1].
@oracle_impl(hardware="B200", point="fe37f3bb", BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0 = inputs
    del _shape_param_0
    n_cols = 128

    start = torch.empty_strided(
        (1, n_cols),
        (n_cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    end = torch.empty_strided(
        (1, n_cols),
        (n_cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    start_amax = torch.empty_strided((1, 1), (1, 1), device=arg0_1.device, dtype=torch.float32)
    start_log = torch.empty_strided((1, 1), (1, 1), device=arg0_1.device, dtype=torch.float32)
    start_valid = torch.empty_strided((1,), (1,), device=arg0_1.device, dtype=torch.bool)
    end_amax = torch.empty_strided((1, 1), (1, 1), device=arg0_1.device, dtype=torch.float32)
    end_log = torch.empty_strided((1, 1), (1, 1), device=arg0_1.device, dtype=torch.float32)
    end_valid = torch.empty_strided((1,), (1,), device=arg0_1.device, dtype=torch.bool)
    loss = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.float32)

    _qa_two_head_train_kernel[(1,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        start,
        end,
        start_amax,
        start_log,
        start_valid,
        end_amax,
        end_log,
        end_valid,
        loss,
        INPUT_STRIDE_ROW=arg0_1.stride(0),
        N_COLS=n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return (
        start,
        end,
        start_amax,
        start_log,
        start_valid,
        end_amax,
        end_log,
        end_valid,
        loss,
    )
