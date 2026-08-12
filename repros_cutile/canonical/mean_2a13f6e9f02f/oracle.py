"""cuTile port of mean_2a13f6e9f02f: SqueezeNet ReLU + 13x13 spatial mean + le mask.

Per-(batch, channel) row kernel: for each of the 32*1000 (batch, channel) rows,
compute NaN-preserving ReLU on 169 spatial elements, spatial mean to bf16, and
`x <= 0` mask (folded to input predicate).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
CHANNELS = 1000
H = 13
W = 13
HW = H * W  # 169
BLOCK_HW = 256


@ct.kernel
def _relu_mean_kernel(
    x_ptr,       # bf16 [BATCH, CHANNELS, HW]
    mean_ptr,    # bf16 [BATCH, CHANNELS]
):
    b = ct.bid(0)
    c = ct.bid(1)

    x = ct.load(
        x_ptr,
        index=(b, c, 0),
        shape=(1, 1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    zero_f = ct.full(shape=(1, 1, BLOCK_HW), fill_value=0.0, dtype=ct.float32)

    # NaN-preserving ReLU
    is_nan = x_f != x_f
    relu = ct.where(is_nan, x_f, ct.where(x_f > zero_f, x_f, zero_f))

    # Mask valid lanes (indices < HW).
    idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    idx_3d = ct.reshape(idx, (1, 1, BLOCK_HW))
    hw_mask = idx_3d < HW
    relu_masked = ct.where(hw_mask, relu, zero_f)
    total = ct.sum(relu_masked)
    mean_val = total * (1.0 / HW)
    mean_bf = ct.astype(mean_val, ct.bfloat16)
    ct.store(mean_ptr, index=(b, c), tile=ct.reshape(mean_bf, (1, 1)))


@oracle_impl(hardware="B200", point="5d063765")
def oracle_forward(inputs):
    x, _shape = inputs
    x_3d = x.view(BATCH, CHANNELS, HW)
    mean_out = torch.empty_strided(
        (BATCH, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, CHANNELS, 1),
        _relu_mean_kernel,
        (x_3d, mean_out),
    )
    # Compute le_mask via torch since cuTile can't store a non-pow2 tile.
    le_mask_out = torch.le(x, x.new_zeros(1))
    return mean_out, le_mask_out
