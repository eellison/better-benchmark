"""cuTile port of amax_sum_sum_5e142a842c9e: Blenderbot bf16 biased cross-entropy.

Structure:
  1. cuTile row kernel: bf16(logits + bias) -> row amax + row denom (fp32
     stable logsumexp) + stored bf16 [rows,K] add tensor.
  2. Torch: target gather from log_softmax then compute the -100 masked loss
     and the mean division. Both are cheap epilogues that don't dominate.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bias_softmax_row_kernel(
    logits_ptr,   # bf16 [rows, K]
    bias_ptr,     # bf16 [K]
    add_ptr,      # bf16 [rows, K]
    amax_ptr,     # f32  [rows]
    denom_ptr,    # f32  [rows]
    K: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    logits = ct.load(logits_ptr, index=(row, 0), shape=(1, BLOCK_N))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,))
    bias_2d = ct.reshape(bias, (1, BLOCK_N))
    added_bf = ct.astype(ct.astype(logits, ct.float32) + ct.astype(bias_2d, ct.float32), ct.bfloat16)
    ct.store(add_ptr, index=(row, 0), tile=added_bf)

    scores = ct.astype(added_bf, ct.float32)
    # Mask out padding: cols >= K -> -inf so they don't affect max/sum.
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    col_ok = ct.reshape(col_idx < K, (1, BLOCK_N))
    minf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    scores = ct.where(col_ok, scores, minf)
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    zero_f = ct.full((1, BLOCK_N), 0.0, dtype=ct.float32)
    numer = ct.where(col_ok, numer, zero_f)
    denom = ct.sum(numer, axis=1, keepdims=True)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@oracle_impl(hardware="B200", point="05e1a070", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N: int):
    labels, logits, bias, shape0, shape1 = inputs
    add_shape = _shape_tuple(shape0)  # [16,128,8008]
    matrix_shape = _shape_tuple(shape1)  # [2048,8008]
    n_rows = int(labels.numel())
    K = int(matrix_shape[1])
    device = logits.device

    # Pick a power-of-two BLOCK_N >= K (default 8192 >= 8008).
    block_n = max(int(BLOCK_N), _next_pow2(K))

    add_bf16 = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty((n_rows,), device=device, dtype=torch.float32)
    denom = torch.empty((n_rows,), device=device, dtype=torch.float32)

    add_2d = add_bf16.view(n_rows, K)
    bias_1d = bias.view(K)
    logits_2d = logits.view(n_rows, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _bias_softmax_row_kernel,
        (logits_2d, bias_1d, add_2d, amax, denom, K, block_n),
    )

    # Torch epilogue: target gather + masked loss + mean division.
    labels_flat = labels.view(-1)
    is_valid = labels_flat != -100
    safe_label = torch.where(is_valid, labels_flat, torch.zeros_like(labels_flat))
    target_logit = torch.gather(logits_2d.float(), 1, safe_label.unsqueeze(1)).squeeze(1)
    target_bias = bias_1d.float().gather(0, safe_label)
    target = (target_logit + target_bias).to(torch.bfloat16).to(torch.float32)
    log_denom = torch.log(denom)
    logp = target - amax - log_denom
    per_row = (-logp).to(torch.bfloat16).to(torch.float32)
    per_row = torch.where(is_valid, per_row, torch.zeros_like(per_row))
    total_loss = per_row.sum().to(torch.bfloat16).to(torch.float32)
    total_valid = is_valid.to(torch.float32).sum().to(torch.bfloat16).to(torch.float32)
    out = (total_loss / total_valid).to(torch.bfloat16)
    return add_bf16, out
