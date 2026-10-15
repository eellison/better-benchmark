"""cuTile port of amax_sum_sum_73d95925b4b4: GPT-J QA train cross-entropy.

Small [1, 128] log-softmax + label gather with ignore-index handling. Two
heads (start, end). Uses cuTile for the row-max/logsumexp/gather; scalar
epilogue done in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 128


@ct.kernel
def _qa_head_kernel(
    clone_ptr,       # bf16 [N_COLS]
    label_ptr,       # i64 [1]
    row_amax_ptr,    # f32 [1]
    row_log_ptr,     # f32 [1]
    valid_ptr,       # bool [1]
    gathered_ptr,    # f32 [1]
):
    x = ct.load(clone_ptr, index=(0,), shape=(N_COLS,))
    x_f = ct.astype(x, ct.float32)
    row_max = ct.max(x_f)
    shifted = x_f - row_max
    log_denom = ct.log(ct.sum(ct.exp(shifted)))
    logp = shifted - log_denom
    logp_bf16 = ct.astype(logp, ct.bfloat16)
    logp_bf16_f = ct.astype(logp_bf16, ct.float32)

    label = ct.load(label_ptr, index=(0,), shape=(1,))
    label_scalar = ct.reshape(label, ())
    zero_i = ct.full(shape=(), fill_value=0, dtype=ct.int64)
    ncols_i = ct.full(shape=(), fill_value=N_COLS, dtype=ct.int64)
    # Emulate clamp(label, 0, N_COLS) with ct.where.
    clamped_lo = ct.where(label_scalar < zero_i, zero_i, label_scalar)
    label_clamped = ct.where(clamped_lo > ncols_i, ncols_i, clamped_lo)
    valid = label_clamped != ncols_i
    safe_label = ct.where(valid, label_clamped, zero_i)

    cols = ct.arange(N_COLS, dtype=ct.int64)
    match = cols == safe_label
    match_f = ct.astype(match, ct.float32)
    gathered = ct.sum(match_f * logp_bf16_f)

    ct.store(row_amax_ptr, index=(0,), tile=ct.reshape(row_max, (1,)))
    ct.store(row_log_ptr, index=(0,), tile=ct.reshape(log_denom, (1,)))
    ct.store(valid_ptr, index=(0,), tile=ct.reshape(valid, (1,)))
    ct.store(gathered_ptr, index=(0,), tile=ct.reshape(gathered, (1,)))


@oracle_impl(hardware="B200", point="fe37f3bb")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0 = inputs
    device = arg0_1.device

    slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -6)
    view = slice_1.view(1, N_COLS, 2)
    start = view[:, :, 0].contiguous()
    end = view[:, :, 1].contiguous()

    start_amax = torch.empty_strided((1, 1), (1, 1), device=device, dtype=torch.float32)
    start_log = torch.empty_strided((1, 1), (1, 1), device=device, dtype=torch.float32)
    start_valid = torch.empty_strided((1,), (1,), device=device, dtype=torch.bool)
    end_amax = torch.empty_strided((1, 1), (1, 1), device=device, dtype=torch.float32)
    end_log = torch.empty_strided((1, 1), (1, 1), device=device, dtype=torch.float32)
    end_valid = torch.empty_strided((1,), (1,), device=device, dtype=torch.bool)
    start_gath = torch.empty((1,), device=device, dtype=torch.float32)
    end_gath = torch.empty((1,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _qa_head_kernel,
              (start.view(-1), arg1_1, start_amax.view(-1), start_log.view(-1), start_valid, start_gath))
    ct.launch(stream, (1, 1, 1), _qa_head_kernel,
              (end.view(-1), arg2_1, end_amax.view(-1), end_log.view(-1), end_valid, end_gath))

    ignored = arg3_1.to(torch.float32)
    start_loss = torch.where(start_valid, -start_gath, ignored)
    end_loss = torch.where(end_valid, -end_gath, ignored)
    start_count = start_valid.to(torch.float32)
    end_count = end_valid.to(torch.float32)
    start_mean = start_loss / start_count
    end_mean = end_loss / end_count
    loss = ((start_mean + end_mean) / 2.0).squeeze()

    return (start, end, start_amax, start_log, start_valid, end_amax, end_log, end_valid, loss)
