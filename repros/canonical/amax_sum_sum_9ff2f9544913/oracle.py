"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J question-answering two-head bf16 cross-entropy scope in one Triton kernel, including the `[128,2]` to `[1,128,2]` view/split, returned contiguous bf16 start/end clones, clamped start/end labels, fp32 stable row amax/libdevice.exp/sum/log for each head, explicit bf16 log-softmax rounding before gather, bf16 valid-loss divisions, and final bf16 average, whereas Inductor lowers the split/clone/clamp/log-softmax/gather/masked-loss/count/div/average graph through generic pointwise, reduction, and copy kernels; Inductor cannot do this today because its pattern library does not canonicalize small two-head QA log_softmax plus clamped-label gather and visible clone side outputs into one full-scope lowering; the fix is NEW_PATTERN: add a guarded QA cross-entropy template that fuses clone stores, online logsumexp, target gather, ignore-index handling, and bf16 scalar epilogue while preserving the returned-output envelope."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _qa_two_head_xent_kernel(
    logits_ptr,
    start_label_ptr,
    end_label_ptr,
    start_out_ptr,
    end_out_ptr,
    loss_out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N_COLS

    start_bf16 = tl.load(logits_ptr + cols * 2, mask=mask, other=0.0).to(tl.bfloat16)
    end_bf16 = tl.load(logits_ptr + cols * 2 + 1, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(start_out_ptr + cols, start_bf16, mask=mask)
    tl.store(end_out_ptr + cols, end_bf16, mask=mask)

    start = start_bf16.to(tl.float32)
    end = end_bf16.to(tl.float32)
    start = tl.where(mask, start, -float("inf"))
    end = tl.where(mask, end, -float("inf"))

    start_max = tl.max(start, axis=0)
    end_max = tl.max(end, axis=0)
    start_shifted = start - start_max
    end_shifted = end - end_max
    start_log_denom = libdevice.log(tl.sum(libdevice.exp(start_shifted), axis=0))
    end_log_denom = libdevice.log(tl.sum(libdevice.exp(end_shifted), axis=0))
    start_logp = (start_shifted - start_log_denom).to(tl.bfloat16)
    end_logp = (end_shifted - end_log_denom).to(tl.bfloat16)

    raw_start_label = tl.load(start_label_ptr)
    raw_end_label = tl.load(end_label_ptr)
    start_label = tl.minimum(tl.maximum(raw_start_label, 0), N_COLS)
    end_label = tl.minimum(tl.maximum(raw_end_label, 0), N_COLS)
    valid_start = start_label != N_COLS
    valid_end = end_label != N_COLS
    safe_start_label = tl.where(valid_start, start_label, 0)
    safe_end_label = tl.where(valid_end, end_label, 0)

    gathered_start = tl.sum(
        tl.where(cols == safe_start_label, start_logp.to(tl.float32), 0.0),
        axis=0,
    )
    gathered_end = tl.sum(
        tl.where(cols == safe_end_label, end_logp.to(tl.float32), 0.0),
        axis=0,
    )

    neg_start = (0.0 - gathered_start).to(tl.bfloat16)
    neg_end = (0.0 - gathered_end).to(tl.bfloat16)
    loss_start = tl.where(valid_start, neg_start, tl.full((), 0.0, tl.float32).to(tl.bfloat16))
    loss_end = tl.where(valid_end, neg_end, tl.full((), 0.0, tl.float32).to(tl.bfloat16))
    count_start = tl.where(valid_start, 1.0, 0.0).to(tl.bfloat16)
    count_end = tl.where(valid_end, 1.0, 0.0).to(tl.bfloat16)

    mean_start = (loss_start.to(tl.float32) / count_start.to(tl.float32)).to(tl.bfloat16)
    mean_end = (loss_end.to(tl.float32) / count_end.to(tl.float32)).to(tl.bfloat16)
    total = (mean_start.to(tl.float32) + mean_end.to(tl.float32)).to(tl.bfloat16)
    result = (total.to(tl.float32) / 2.0).to(tl.bfloat16)
    tl.store(loss_out_ptr, result)


# eb172ec5: GPT-J QA bf16 logits [128,2], start/end labels i64[1].
@oracle_impl(hardware="B200", point="eb172ec5", BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs
    del _shape_param_0

    start = torch.empty_strided(
        (1, int(arg0_1.shape[0])),
        (int(arg0_1.shape[0]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    end = torch.empty_strided(
        (1, int(arg0_1.shape[0])),
        (int(arg0_1.shape[0]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    loss = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.bfloat16)

    _qa_two_head_xent_kernel[(1,)](
        arg0_1,
        arg1_1,
        arg2_1,
        start,
        end,
        loss,
        N_COLS=int(arg0_1.shape[0]),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return start, end, loss
