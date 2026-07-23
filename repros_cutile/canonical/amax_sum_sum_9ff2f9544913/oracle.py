"""cuTile port of amax_sum_sum_9ff2f9544913: GPT-J QA two-head cross-entropy.

Compute log-softmax for each of 128 x 2 logits (start/end heads), gather at
clamped label, output bf16 (start_logp_flat, end_logp_flat) plus mean bf16
loss. cuTile does the per-head log_softmax + gather; torch handles the label
clamp and final scalar arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 128


@ct.kernel
def _log_softmax_gather_kernel(
    logits_ptr,  # bf16 [N_COLS]
    label_ptr,   # i64 [1]  (safe label already clamped and remapped)
    valid_ptr,   # b8 [1]
    out_ptr,     # bf16 [N_COLS]  (log_softmax result)
    loss_ptr,    # bf16 [1]       (-log_softmax[label] * valid)
    N: ct.Constant[int],
):
    tid = ct.bid(0)  # only 1 tile, tid always 0
    x_bf = ct.load(logits_ptr, index=(0,), shape=(N,))
    x = ct.astype(x_bf, ct.float32)
    xmax = ct.max(x, axis=0, keepdims=True)
    shifted = x - xmax
    denom = ct.sum(ct.exp(shifted), axis=0, keepdims=True)
    log_denom = ct.log(denom)
    logp = shifted - log_denom
    logp_bf = ct.astype(logp, ct.bfloat16)
    ct.store(out_ptr, index=(0,), tile=logp_bf)

    # Gather at the label index — implement via ct.where(cols == label, logp, 0) + sum
    label_i64 = ct.load(label_ptr, index=(0,), shape=(1,))
    valid_bool = ct.load(valid_ptr, index=(0,), shape=(1,))
    cols = ct.astype(ct.arange(N, dtype=ct.int32), ct.int64)
    label_broadcast = ct.reshape(label_i64, (1,))  # keep shape (1,)
    match = cols == label_broadcast
    gathered = ct.sum(ct.where(match, ct.astype(logp_bf, ct.float32),
                                ct.full((N,), 0.0, dtype=ct.float32)),
                       axis=0, keepdims=True)
    neg = ct.astype(0.0 - gathered, ct.bfloat16)
    zero_bf = ct.full((1,), 0.0, dtype=ct.bfloat16)
    loss = ct.where(valid_bool, neg, zero_bf)
    ct.store(loss_ptr, index=(0,), tile=loss)


@oracle_impl(hardware="B200", point="eb172ec5", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs
    del _shape_param_0
    device = arg0_1.device

    # Extract start/end logits: arg0_1 is [128, 2]. clone rows contiguous.
    start_bf = arg0_1[:, 0].contiguous().view(1, 128)  # [1, 128]
    end_bf = arg0_1[:, 1].contiguous().view(1, 128)

    # Compute clamped labels + valid mask
    clamp_start = torch.clamp(arg1_1, 0, N_COLS)
    clamp_end = torch.clamp(arg2_1, 0, N_COLS)
    valid_start = (clamp_start != N_COLS)
    valid_end = (clamp_end != N_COLS)
    safe_start = torch.where(valid_start, clamp_start, torch.zeros_like(clamp_start))
    safe_end = torch.where(valid_end, clamp_end, torch.zeros_like(clamp_end))

    # log_softmax + gather in cuTile; we don't need to keep the logp
    _logp_start = torch.empty(128, device=device, dtype=torch.bfloat16)
    _logp_end = torch.empty(128, device=device, dtype=torch.bfloat16)
    loss_start = torch.empty(1, device=device, dtype=torch.bfloat16)
    loss_end = torch.empty(1, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (1, 1, 1), _log_softmax_gather_kernel,
        (start_bf.view(128), safe_start, valid_start, _logp_start, loss_start, 128),
    )
    ct.launch(
        stream, (1, 1, 1), _log_softmax_gather_kernel,
        (end_bf.view(128), safe_end, valid_end, _logp_end, loss_end, 128),
    )

    # Compute count-based mean loss and final average
    count_start_bf = valid_start.to(torch.bfloat16).sum().view(())
    count_end_bf = valid_end.to(torch.bfloat16).sum().view(())
    mean_start = loss_start.view(()) / count_start_bf
    mean_end = loss_end.view(()) / count_end_bf
    total = mean_start + mean_end
    result = total / torch.tensor(2.0, dtype=torch.bfloat16, device=device)

    # Return: clone (bf16 copy of start row), clone_1 (bf16 copy of end row), loss.
    return start_bf.clone(), end_bf.clone(), result
