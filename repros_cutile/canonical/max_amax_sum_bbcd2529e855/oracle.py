"""cuTile port of max_amax_sum_bbcd2529e855: GPT-OSS causal-mask virtual-cat softmax.

Ports the Triton `_causal_virtual_cat_softmax_kernel`. Each `query` row does:
  - Build a triangular `where` mask (bf16 side output).
  - Load channel scores, scale by 0.125, add mask bias.
  - Include a per-channel "extra column" as the virtual (k+1)-th column.
  - Two-pass max reduction to match the Repro (max then amax).
  - Softmax over CAT_LEN cols; drop the extra column when storing.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


MASK_NEG = -3.3895313892515355e38


@ct.kernel
def _causal_virtual_cat_softmax_kernel(
    x_arr,      # (CHANNELS, Q_LEN, K_LEN) bf16
    extra_arr,  # (CHANNELS,) bf16
    where_arr,  # (Q_LEN, K_LEN) bf16
    out_arr,    # (CHANNELS, Q_LEN, K_LEN) bf16
    K_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    channel = ct.bid(0)
    query = ct.bid(1)

    # Build (1, BLOCK_N) column indices; keep is (query >= col_index).
    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    # Convert query (int) to a broadcastable value.
    keep = ct.less_equal(cols, ct.full(shape=(BLOCK_N,), fill_value=query, dtype=ct.int32))
    mask_bias = ct.where(
        keep,
        ct.full(shape=(BLOCK_N,), fill_value=0.0, dtype=ct.float32),
        ct.full(shape=(BLOCK_N,), fill_value=MASK_NEG, dtype=ct.float32),
    )
    # Store where_arr[query, :] as bf16 (only for channel==0).
    if channel == 0:
        # Reshape mask_bias to (1, BLOCK_N) for a 2D store.
        mb_bf16 = ct.astype(mask_bias, ct.bfloat16)
        ct.store(
            where_arr,
            index=(query, 0),
            tile=ct.reshape(mb_bf16, (1, BLOCK_N)),
        )

    # Load data (channel, query, :) as (1, 1, BLOCK_N).
    x = ct.load(x_arr, index=(channel, query, 0), shape=(1, 1, BLOCK_N))
    x_1d = ct.reshape(x, (BLOCK_N,))
    x_f = ct.astype(x_1d, ct.float32)
    data_scores = x_f * 0.125 + mask_bias

    extra = ct.load(extra_arr, index=(channel,), shape=(1,))
    extra_f = ct.astype(extra, ct.float32)
    # Replace last "col K_LEN" (the extra virtual col) with extra_f value.
    # cols == K_LEN means it's the extra col; but if BLOCK_N == K_LEN then no
    # extra col; if BLOCK_N > K_LEN, the extra col is at index K_LEN.
    is_extra = ct.equal(cols, ct.full(shape=(BLOCK_N,), fill_value=K_LEN, dtype=ct.int32))
    # Broadcast extra_f (shape (1,)) to (BLOCK_N,).
    extra_bcast = ct.reshape(extra_f, (1,))
    # Instead of broadcasting shapes, use where's scalar semantics.
    scores = ct.where(
        ct.less(cols, ct.full(shape=(BLOCK_N,), fill_value=K_LEN, dtype=ct.int32)),
        data_scores,
        ct.where(
            is_extra,
            # extract scalar via full-broadcast
            ct.full(shape=(BLOCK_N,), fill_value=0.0, dtype=ct.float32) + extra_f,
            ct.full(shape=(BLOCK_N,), fill_value=-float("inf"), dtype=ct.float32),
        ),
    )

    # First max (across CAT_LEN = K_LEN+1 valid cols).
    first_max = ct.max(scores)
    shifted = scores - first_max
    # Mask out invalid cols (>K_LEN).
    shifted = ct.where(
        ct.less_equal(cols, ct.full(shape=(BLOCK_N,), fill_value=K_LEN, dtype=ct.int32)),
        shifted,
        ct.full(shape=(BLOCK_N,), fill_value=-float("inf"), dtype=ct.float32),
    )
    second_max = ct.max(shifted)
    numer = ct.exp(shifted - second_max)
    numer = ct.where(
        ct.less_equal(cols, ct.full(shape=(BLOCK_N,), fill_value=K_LEN, dtype=ct.int32)),
        numer,
        ct.full(shape=(BLOCK_N,), fill_value=0.0, dtype=ct.float32),
    )
    denom = ct.sum(numer)
    probs = numer / denom
    probs_bf16 = ct.astype(probs, ct.bfloat16)
    # Store only [0..K_LEN) columns as bf16 output.
    ct.store(
        out_arr,
        index=(channel, query, 0),
        tile=ct.reshape(probs_bf16, (1, 1, BLOCK_N)),
    )


# d9b19a63: GPT-OSS bf16 generated causal mask plus virtual extra-column softmax.
@oracle_impl(hardware="B200", point="d9b19a63", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, shape4 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    channels = int(arg0_1.shape[0])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    out_shape = tuple(int(dim) for dim in shape4)

    where = torch.empty_strided(
        (1, 1, q_len, k_len),
        (q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # where is [1, 1, q_len, k_len]; treat as [q_len, k_len] for the store.
    where_2d = where.view(q_len, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, q_len, 1),
        _causal_virtual_cat_softmax_kernel,
        (arg0_1, arg1_1, where_2d, out, k_len, BLOCK_N),
    )
    return where, out
