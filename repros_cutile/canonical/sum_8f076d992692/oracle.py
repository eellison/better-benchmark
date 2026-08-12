"""cuTile port of sum_8f076d992692: Demucs channel-slice sum.

For each (batch, pair, col) in the [8,2,382788] output:
  sum_out[b, p, c] = sum of input channels 1..4 at [b, p, c] converted to fp32,
                     then rounded to bf16.
Also returns a metadata-only as_strided view of the input.

The sum-of-4-channels is done INSIDE the cuTile kernel to match Triton's
kernel structure (which reads c1..c4 in-kernel and stores their sum).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS_IN = 5
PAIR = 2
IN_WIDTH = 426888
OUT_WIDTH = 382788
CHANNEL_STRIDE = PAIR * IN_WIDTH
BATCH_STRIDE = CHANNELS_IN * CHANNEL_STRIDE
OUT_ROWS = BATCH * PAIR
ROW_BLOCK = 4
BLOCK_N = 512


@ct.kernel
def _slice_sum_kernel(
    x_ptr,           # bf16 [BATCH, 4, PAIR, IN_WIDTH]  (channels 1..4 view)
    out_ptr,         # bf16 [BATCH, PAIR, OUT_WIDTH]
    OUT_WIDTH_C: ct.Constant[int],
    BLOCK_N_C: ct.Constant[int],
):
    # Grid: (BATCH, PAIR, cdiv(OUT_WIDTH, BLOCK_N))
    b = ct.bid(0)
    p = ct.bid(1)
    col_block = ct.bid(2)

    # Load a (1, 4, 1, BLOCK_N) tile — all 4 channels at (b, :, p, col_block).
    # x is strided so index=(b, 0, p, col_block) with shape (1, 4, 1, BLOCK_N).
    x = ct.load(
        x_ptr, index=(b, 0, p, col_block), shape=(1, 4, 1, BLOCK_N_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    xf = ct.astype(x, ct.float32)
    # Reduce over the 4-channel axis (axis=1)
    total = ct.sum(xf, axis=1)  # shape (1, 1, BLOCK_N)
    total_bf = ct.astype(total, ct.bfloat16)

    # Column mask (BLOCK_N might extend past OUT_WIDTH)
    cols = ct.arange(BLOCK_N_C, dtype=ct.int32) + col_block * BLOCK_N_C
    col_mask = cols < OUT_WIDTH_C

    total_1d = ct.reshape(total_bf, (BLOCK_N_C,))
    b_idx = ct.full((BLOCK_N_C,), b, dtype=ct.int32)
    p_idx = ct.full((BLOCK_N_C,), p, dtype=ct.int32)
    ct.scatter(out_ptr, (b_idx, p_idx, cols), total_1d, mask=col_mask)


@oracle_impl(hardware="B200", point="0dd51176")
def oracle_forward(inputs):
    (arg0_1,) = inputs
    # Metadata-only view: channels 1..4, first OUT_WIDTH cols per pair-slot.
    slice_out = torch.as_strided(
        arg0_1,
        (BATCH, 4, PAIR, OUT_WIDTH),
        (BATCH_STRIDE, CHANNEL_STRIDE, IN_WIDTH, 1),
        CHANNEL_STRIDE,
    )
    sum_out = torch.empty_strided(
        (BATCH, PAIR, OUT_WIDTH),
        (PAIR * OUT_WIDTH, OUT_WIDTH, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, PAIR, ct.cdiv(OUT_WIDTH, BLOCK_N)),
        _slice_sum_kernel,
        (slice_out, sum_out, OUT_WIDTH, BLOCK_N),
    )
    return slice_out, sum_out
