"""cuTile port of max_amax_sum_66e6dc6d2131: gpt-oss softmax with virtual-cat sink.

For each (n, c, r) row of length 1000:
- x = arg0 * 0.125 + arg1  (bf16)
- augmented row = concat([x, sink_c])  # sink is arg2[c] broadcast
- softmax over length 1001
- output = softmax[:, :, :, :1000]  # drop the sink column
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


K = 1000
K_PAD = 1024  # next_pow2 covering K+1=1001


@ct.kernel
def _sink_softmax_kernel(
    x_ptr,          # bf16 [rows, K]
    mask_ptr,       # bf16 [rows, K]  arg1 broadcast (per (r,c) row).
    sink_ptr,       # bf16 [rows] per-channel sink value (broadcasted)
    out_ptr,        # bf16 [rows, K]  softmax result (dropped last column)
    K_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_K),
                padding_mode=ct.PaddingMode.ZERO)
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_K),
                   padding_mode=ct.PaddingMode.ZERO)
    sink = ct.load(sink_ptr, index=(row,), shape=(1,))

    # x_val = (x * 0.125) + mask   (bf16 add/mul in bf16)
    x_scaled = ct.astype(ct.astype(x, ct.float32) * 0.125, ct.bfloat16)
    x_added = ct.astype(
        ct.astype(x_scaled, ct.float32) + ct.astype(mask, ct.float32),
        ct.bfloat16)

    # Build augmented row: for cols in [0, K_), x_added; for cols == K_, sink; else 0.
    cols_i = ct.arange(BLOCK_K, dtype=ct.int32)
    col_lt_K = ct.reshape(cols_i < K_, (1, BLOCK_K))
    col_eq_K = ct.reshape(cols_i == K_, (1, BLOCK_K))

    sink_bc = ct.reshape(sink, (1, 1))
    zero_bf = ct.full((1, BLOCK_K), 0.0, dtype=ct.bfloat16)
    cat_val = ct.where(col_lt_K, x_added, ct.where(col_eq_K, sink_bc, zero_bf))

    # For softmax masking of OOB (col > K), use -inf so exp=0.
    neg_inf_f = ct.full((1, BLOCK_K), -1.0e30, dtype=ct.float32)
    valid = col_lt_K | col_eq_K

    max_bf = ct.max(cat_val, axis=1, keepdims=True)
    sub_bf = cat_val - max_bf
    sub_f = ct.astype(sub_bf, ct.float32)
    sub_f = ct.where(valid, sub_f, neg_inf_f)
    amax = ct.max(sub_f, axis=1, keepdims=True)
    sub_from_amax = sub_f - amax
    exp_v = ct.exp(sub_from_amax)
    zero_f = ct.full((1, BLOCK_K), 0.0, dtype=ct.float32)
    exp_v = ct.where(valid, exp_v, zero_f)
    sum_v = ct.sum(exp_v, axis=1, keepdims=True)
    div_v = exp_v / sum_v
    div_bf = ct.astype(div_v, ct.bfloat16)

    # Output = div_bf[:, :K_] — cuTile store must be full-tile; we pad.
    # Write to a K_PAD-strided output; wrapper will slice back to K.
    ct.store(out_ptr, index=(row, 0), tile=div_bf)


@oracle_impl(hardware="B200", point="1bcb4709")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, *_shape) = inputs
    device = arg0_1.device
    # arg0 [64,1000,1000] = 1*64*1000 rows of K=1000. arg1 [1,1,1000,1000] broadcast to same.
    # arg2 [64] one sink per channel.
    total_rows = 64 * 1000

    # view = arg0.view(1, 64, 1000, 1000) → [rows, K]
    x = arg0_1.view(64, 1000, 1000).contiguous().view(total_rows, K)
    # arg1 broadcasts: shape [1,1,1000,1000] → [64, 1000, 1000] per (n,c,r,k).
    mask = arg1_1.expand(1, 64, 1000, 1000).contiguous().view(total_rows, K)
    # sink per row: arg2[c] duplicated across all rows. rows = 64*1000, arg2 has 64.
    sink = arg2_1.view(64, 1, 1).expand(64, 1000, 1).contiguous().view(total_rows)

    out_pad = torch.empty((total_rows, K_PAD), device=device, dtype=torch.bfloat16)

    # Pad inputs to K_PAD
    x_pad = torch.zeros((total_rows, K_PAD), device=device, dtype=torch.bfloat16)
    x_pad[:, :K].copy_(x)
    mask_pad = torch.zeros((total_rows, K_PAD), device=device, dtype=torch.bfloat16)
    mask_pad[:, :K].copy_(mask)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (total_rows, 1, 1), _sink_softmax_kernel,
              (x_pad, mask_pad, sink, out_pad, K, K_PAD))

    out = out_pad[:, :K].contiguous().view(1, 64, 1000, K)
    view_2 = out.view(64, 1000, K)
    return view_2
