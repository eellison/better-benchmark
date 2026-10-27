"""cuTile port of max_amax_sum_40ecfc34136c: GPT-OSS sliding-window softmax.

The sliding-window mask depends only on (q, k) index arithmetic, not on the
input tensor, so it is materialized once in torch. The cuTile kernel handles
the softmax over (channel_block, query) tiles, folding in the mask and the
virtual per-channel sentinel column in the softmax denominator.

Bit-cast bf16->f32 rounding in Triton is emulated by
`astype(x_f32, bfloat16)` followed by `astype(_, float32)`, which produces
the same round-trip effect (cuTile's astype uses RTNE by default).

Grid: (Q_LEN, cdiv(CHANNELS, BLOCK_C)). BLOCK_N=1024 pads past K_LEN=1000.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NEG_INF_BF16 = -3.3895313892515355e38
WINDOW = 128


@ct.kernel
def _sliding_window_softmax_kernel(
    x_ptr,               # bf16 [CHANNELS, Q_LEN, K_LEN]  (flat storage)
    extra_ptr,           # bf16 [CHANNELS]
    where_ptr,           # bf16 [Q_LEN, K_LEN]           (flat storage)
    out_ptr,             # bf16 [CHANNELS, Q_LEN, K_LEN]  (flat storage)
    CHANNELS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    query = ct.bid(0)
    channel_block = ct.bid(1)

    channels = ct.arange(BLOCK_C, dtype=ct.int64) + channel_block * BLOCK_C
    cols = ct.arange(BLOCK_N, dtype=ct.int64)
    channels_2d = ct.reshape(channels, (BLOCK_C, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    channels_broad = ct.broadcast_to(channels_2d, (BLOCK_C, BLOCK_N))
    cols_broad = ct.broadcast_to(cols_2d, (BLOCK_C, BLOCK_N))

    channel_mask = channels_broad < CHANNELS
    data_col_mask = cols_broad < K_LEN
    valid_col_mask = cols_broad < (K_LEN + 1)
    data_mask = channel_mask & data_col_mask
    valid_mask = channel_mask & valid_col_mask

    # Load where mask (Q_LEN, K_LEN) — same for every channel
    where_offsets_1d = query * K_LEN + cols
    safe_where_offsets = ct.where(
        cols < K_LEN, where_offsets_1d, ct.zeros((BLOCK_N,), dtype=ct.int64)
    )
    where_row_bf = ct.gather(where_ptr, safe_where_offsets)  # (BLOCK_N,)
    where_row_2d = ct.reshape(where_row_bf, (1, BLOCK_N))
    where_broad = ct.broadcast_to(where_row_2d, (BLOCK_C, BLOCK_N))

    # Load x — contiguous stride (Q_LEN*K_LEN, K_LEN, 1)
    x_offsets = (
        channels_broad * (Q_LEN * K_LEN)
        + query * K_LEN
        + cols_broad
    )
    safe_x_offsets = ct.where(
        data_mask, x_offsets, ct.zeros((BLOCK_C, BLOCK_N), dtype=ct.int64)
    )
    x_bf = ct.gather(x_ptr, safe_x_offsets)
    x = ct.astype(x_bf, ct.float32)

    # scaled = round-bf16(x * 0.125)
    scaled = ct.astype(ct.astype(x * 0.125, ct.bfloat16), ct.float32)
    # data_scores = round-bf16(scaled + where_bias)
    where_f = ct.astype(where_broad, ct.float32)
    data_scores = ct.astype(ct.astype(scaled + where_f, ct.bfloat16), ct.float32)

    # extra[channels] — 1D gather
    safe_extra_channels = ct.where(
        channels < CHANNELS, channels, ct.zeros((BLOCK_C,), dtype=ct.int64)
    )
    extra_bf = ct.gather(extra_ptr, safe_extra_channels)
    extra = ct.astype(extra_bf, ct.float32)  # (BLOCK_C,)
    extra_2d = ct.reshape(extra, (BLOCK_C, 1))
    extra_broad = ct.broadcast_to(extra_2d, (BLOCK_C, BLOCK_N))

    # For cols < K_LEN: data_scores; for cols == K_LEN: extra; else -inf
    scores = ct.where(cols_broad < K_LEN, data_scores, extra_broad)
    neg_inf = ct.full((BLOCK_C, BLOCK_N), -float("inf"), dtype=ct.float32)
    scores = ct.where(valid_mask, scores, neg_inf)

    first_max = ct.max(scores, axis=1)  # (BLOCK_C,)
    first_max_2d = ct.reshape(first_max, (BLOCK_C, 1))
    first_max_broad = ct.broadcast_to(first_max_2d, (BLOCK_C, BLOCK_N))
    shifted = scores - first_max_broad
    shifted = ct.where(valid_mask, shifted, neg_inf)

    second_max = ct.max(shifted, axis=1)
    second_max_2d = ct.reshape(second_max, (BLOCK_C, 1))
    second_max_broad = ct.broadcast_to(second_max_2d, (BLOCK_C, BLOCK_N))
    numer = ct.exp(shifted - second_max_broad)
    numer = ct.where(
        valid_mask, numer, ct.zeros((BLOCK_C, BLOCK_N), dtype=ct.float32)
    )
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_C, 1))
    denom_broad = ct.broadcast_to(denom_2d, (BLOCK_C, BLOCK_N))
    probs = numer / denom_broad

    # Scatter probs back to out for cols < K_LEN
    ct.scatter(
        out_ptr,
        ct.where(data_mask, x_offsets, ct.zeros((BLOCK_C, BLOCK_N), dtype=ct.int64)),
        ct.astype(probs, ct.bfloat16),
        mask=data_mask,
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="d9b19a63", BLOCK_C=2, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_N: int):
    arg0_1, arg1_1, *_ = inputs

    channels = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    device = arg0_1.device

    # Build the sliding-window mask in torch (it doesn't depend on x)
    q_pos = torch.arange(q_len, device=device).unsqueeze(1)  # [Q, 1]
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)  # [1, K]
    window_mask = (k_pos > (q_pos - WINDOW)) & (k_pos <= q_pos)  # [Q, K]
    where_2d = torch.where(
        window_mask,
        torch.tensor(0.0, dtype=torch.bfloat16, device=device),
        torch.tensor(NEG_INF_BF16, dtype=torch.bfloat16, device=device),
    )
    # Materialize per Repro's shape (1,1,Q,K) with weird stride
    where = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    where.view(q_len, k_len).copy_(where_2d)

    out = torch.empty_strided(
        (channels, q_len, k_len),
        _contiguous_stride((channels, q_len, k_len)),
        device=device,
        dtype=torch.bfloat16,
    )

    # Flatten pointer views
    x_flat = arg0_1.reshape(-1)
    extra_flat = arg1_1.reshape(-1)
    where_flat = where.view(-1)
    out_flat = out.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (q_len, (channels + BLOCK_C - 1) // BLOCK_C, 1),
        _sliding_window_softmax_kernel,
        (x_flat, extra_flat, where_flat, out_flat,
         channels, q_len, k_len, BLOCK_C, BLOCK_N),
    )
    return where, out
