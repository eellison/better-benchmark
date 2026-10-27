"""cuTile port of amax_sum_8002af197c08: ConvBERT bias-add softmax over width-9.

For each row (98304 rows), load PADDED=16 lanes via ct.gather (WIDTH=9 valid),
apply row softmax with -inf padding, and scatter back the 9 valid probs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


WIDTH = 9
CHANNELS = 54
PADDED = 16  # power-of-two >= 9


@ct.kernel
def _bias_softmax_single_row_kernel(
    x_flat_ptr,     # (N_ROWS * WIDTH,) bf16
    bias_ptr,       # (CHANNELS,) bf16
    out_flat_ptr,   # (N_ROWS * WIDTH,) bf16
    N_ROWS: ct.Constant[int],
):
    row = ct.bid(0)
    lane = ct.arange(PADDED, dtype=ct.int32)
    lane_mask = lane < WIDTH
    offsets = row * WIDTH + lane  # (PADDED,)

    x = ct.gather(x_flat_ptr, offsets, mask=lane_mask, padding_value=0.0)
    bias_off = offsets % CHANNELS
    b = ct.gather(bias_ptr, bias_off, mask=lane_mask, padding_value=0.0)

    x_f = ct.astype(x, ct.float32) + ct.astype(b, ct.float32)
    neg_inf = ct.full((PADDED,), float("-inf"), dtype=ct.float32)
    scores = ct.where(lane_mask, x_f, neg_inf)

    row_max = ct.max(scores)
    numer = ct.exp(scores - row_max)
    zero_f = ct.full((PADDED,), 0.0, dtype=ct.float32)
    numer_m = ct.where(lane_mask, numer, zero_f)
    denom = ct.sum(numer_m)
    probs = numer_m / denom
    probs_bf16 = ct.astype(probs, ct.bfloat16)

    ct.scatter(out_flat_ptr, offsets, probs_bf16, mask=lane_mask)


@oracle_impl(hardware="B200", point="5fad102b", BLOCK_M=128, BLOCK_N=16)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    x, bias, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_2)
    n_rows = int(out_shape[0])
    out = torch.empty_strided(out_shape, (9, 1, 1), device=x.device, dtype=torch.bfloat16)

    numel = n_rows * WIDTH
    x_flat = x.reshape(numel)
    out_flat = out.reshape(numel)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _bias_softmax_single_row_kernel,
        (x_flat, bias, out_flat, n_rows),
    )
    return out
