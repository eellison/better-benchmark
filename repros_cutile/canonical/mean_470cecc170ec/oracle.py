"""cuTile port of mean_470cecc170ec: GhostNet BN + ReLU + 7x7 spatial mean.

Input x is bf16 channels-last [B=512, C=960, H=7, W=7]. For each (b, c):
  y = ((x[b, c, :, :] - running_mean[c]) / sqrt(running_var[c] + 1e-5)) * weight[c] + bias[c]
  y_bf16 = y.to(bf16); relu = max(y_bf16, 0); out[b, c, 0, 0] = mean(relu.to(f32)) / 49
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = 49
BLOCK_HW = 64  # next power of 2 >= HW=49


@ct.kernel
def _bn_relu_mean_kernel(
    running_mean_ptr,  # bf16 [C]
    x_ptr,             # bf16 [B, C, H, W] channels-last
    running_var_ptr,   # bf16 [C]
    weight_ptr,        # bf16 [C]
    bias_ptr,          # bf16 [C]
    out_ptr,           # bf16 [B, C, 1, 1] -> viewed as [B, C]
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    HW_SIZE: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)

    # H=W=7 aren't powers of 2; pad to HP=WP=8 with ZERO.
    HP: ct.Constant[int] = 8
    WP: ct.Constant[int] = 8

    x_tile = ct.load(x_ptr, index=(b, c, 0, 0), shape=(1, 1, HP, WP),
                     padding_mode=ct.PaddingMode.ZERO)
    x_tile = ct.reshape(x_tile, (HP, WP))
    x_f = ct.astype(x_tile, ct.float32)

    # Load per-channel scalars using shape=(1,)
    mean_c = ct.load(running_mean_ptr, index=(c,), shape=(1,))
    var_c = ct.load(running_var_ptr, index=(c,), shape=(1,))
    w_c = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_c = ct.load(bias_ptr, index=(c,), shape=(1,))
    mean_f = ct.astype(mean_c, ct.float32)
    var_f = ct.astype(var_c, ct.float32)
    w_f = ct.astype(w_c, ct.float32)
    bias_f = ct.astype(bias_c, ct.float32)

    invstd = ct.rsqrt(var_f + 1.0e-5)
    centered = x_f - mean_f
    normalized = centered * invstd
    scaled = normalized * w_f
    affine = scaled + bias_f
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.full(shape=(HP, WP), fill_value=0.0, dtype=ct.bfloat16)
    relu = ct.where(affine_bf16 < zero_bf, zero_bf, affine_bf16)

    # Mask OOB (h >= H or w >= W) to 0 before summing.
    h_idx = ct.arange(HP, dtype=ct.int32)
    w_idx = ct.arange(WP, dtype=ct.int32)
    h_valid = ct.reshape(h_idx < H, (HP, 1))
    w_valid = ct.reshape(w_idx < W, (1, WP))
    valid = h_valid & w_valid
    zero_f = ct.full(shape=(HP, WP), fill_value=0.0, dtype=ct.float32)
    relu_f = ct.where(valid, ct.astype(relu, ct.float32), zero_f)
    total = ct.sum(relu_f)
    pooled = total * (1.0 / HW_SIZE)

    ct.store(out_ptr, index=(b, c), tile=ct.reshape(ct.astype(pooled, ct.bfloat16), (1, 1)))


@oracle_impl(hardware="B200", point="3e244c1d", BLOCK_ROWS=64, BLOCK_HW=BLOCK_HW)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_HW):
    running_mean, x, running_var, weight, bias, _shape, _stride = inputs
    out = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, CHANNELS, CHANNELS),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # x has shape [B, C, H, W] channels-last, cuTile respects the strides.
    out_2d = out.view(BATCH, CHANNELS)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, CHANNELS, 1),
        _bn_relu_mean_kernel,
        (running_mean, x, running_var, weight, bias, out_2d,
         CHANNELS, HEIGHT, WIDTH, HW, BLOCK_HW),
    )
    return out
