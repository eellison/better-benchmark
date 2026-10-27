"""cuTile port of amax_sum_sum_fb4ae0eb915f: DistillGPT2 shifted cross-entropy.

cuTile handles per-row log-softmax reduction (amax + logsumexp). Torch does
the shift/pad, target gather, and mean-over-valid reduction (these use
graph-capturable aten ops).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
VOCAB = 50257
BLOCK_H = 65536  # ceil next-pow2(50257)


@ct.kernel
def _log_softmax_stats_kernel(
    x_ptr,       # f32 [ROWS, BLOCK_H] (zero-padded past VOCAB)
    amax_ptr,    # f32 [ROWS]
    log_ptr,     # f32 [ROWS]
    VOCAB_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    valid = ct.reshape(idx < VOCAB_, (1, BLOCK_H_))
    # OOB set to -inf so it doesn't affect amax.
    x_masked = ct.where(valid, x, float("-inf"))
    row_max = ct.max(x_masked)
    shifted = x - row_max
    # For exp, use masked variant so OOB doesn't contribute.
    exp_val = ct.exp(shifted)
    exp_masked = ct.where(valid, exp_val, 0.0)
    denom = ct.sum(exp_masked)
    log_denom = ct.log(denom)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), row_max, dtype=ct.float32), (1,)))
    ct.store(log_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), log_denom, dtype=ct.float32), (1,)))


@oracle_impl(hardware="B200", point="c1ee6cd0")
def oracle_forward(inputs):
    arg0_1, arg1_1, view_shape = inputs
    device = arg0_1.device

    x_f32 = arg0_1.to(torch.float32)
    x_rows = x_f32.view(ROWS, VOCAB)
    # Pad to BLOCK_H
    x_padded = torch.zeros((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    x_padded.narrow(1, 0, VOCAB).copy_(x_rows)

    amax_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    log_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _log_softmax_stats_kernel,
        (x_padded, amax_1d, log_1d, VOCAB, BLOCK_H),
    )

    amax = amax_1d.view(ROWS, 1)
    log = log_1d.view(ROWS, 1)

    # Torch handles the pad + shift + gather epilogue.
    constant_pad_nd = torch.nn.functional.pad(arg1_1, [0, 1], value=-100)
    slice_1 = constant_pad_nd[:, 1:]
    clone = slice_1.contiguous()
    view_1 = clone.view(-1)

    sub = x_rows - amax  # f32[ROWS, VOCAB]
    sub_1 = sub - log
    ne = view_1 != -100
    zero_i = torch.zeros((), dtype=torch.int64, device=device)
    where = torch.where(ne, view_1, zero_i)
    gather = sub_1.gather(1, where.unsqueeze(1))
    squeeze = gather.squeeze(1)
    neg = -squeeze
    zero_f = torch.zeros((), dtype=torch.float32, device=device)
    where_1 = torch.where(ne, neg, zero_f)
    sum_2 = ne.sum()
    n_valid_f = sum_2.to(torch.float32)
    sum_3 = where_1.sum()
    div = sum_3 / n_valid_f

    return constant_pad_nd, amax, log, n_valid_f, div
