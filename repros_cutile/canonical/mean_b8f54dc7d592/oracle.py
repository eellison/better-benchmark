"""cuTile port of mean_b8f54dc7d592: MobileNetV3 BN + hard-swish + 7x7 spatial mean.

Input x is bf16 channels-last [B=512, C=960, H=7, W=7] (HW=49). For each (b, c):
  affine = ((x - mean[c]) / sqrt(var[c] + 1e-5)) * weight[c] + bias[c]  (fp32)
  affine_bf16 = affine.to(bf16).to(f32)   (bf16 round-trip)
  hardswish = affine_bf16 * clamp(affine_bf16 + 3, 0, 6) / 6
  y = hardswish.to(bf16).to(f32)
  out[b, c] = mean(y) / 49
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
CHANNELS = 960
HW = 49
H = 7
W = 7
HP = 8
WP = 8
EPS = 1.0e-5


@ct.kernel
def _bn_hardswish_spatial_mean_kernel(
    running_mean_ptr,  # bf16 [C]
    x_ptr,             # bf16 [B, C, H, W] channels-last
    running_var_ptr,   # bf16 [C]
    weight_ptr,        # bf16 [C]
    bias_ptr,          # bf16 [C]
    out_ptr,           # bf16 viewed as [B, C]
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    HP: ct.Constant[int],
    WP: ct.Constant[int],
    HW_F: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)

    x_tile = ct.load(x_ptr, index=(b, c, 0, 0), shape=(1, 1, HP, WP),
                     padding_mode=ct.PaddingMode.ZERO)
    x_tile = ct.reshape(x_tile, (HP, WP))
    x_f = ct.astype(x_tile, ct.float32)

    mean_c = ct.astype(ct.load(running_mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var_c = ct.astype(ct.load(running_var_ptr, index=(c,), shape=(1,)), ct.float32)
    w_c = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias_c = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)

    sqrt_var = ct.sqrt(var_c + EPS)
    invstd = 1.0 / sqrt_var
    centered = x_f - mean_c
    normalized = centered * invstd
    affine = normalized * w_c + bias_c
    # bf16 round-trip
    affine_bf16 = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    relu6 = affine_bf16 + 3.0
    zero_f = ct.full(shape=(HP, WP), fill_value=0.0, dtype=ct.float32)
    six_f = ct.full(shape=(HP, WP), fill_value=6.0, dtype=ct.float32)
    relu6 = ct.where(relu6 < 0.0, zero_f, relu6)
    relu6 = ct.where(relu6 > 6.0, six_f, relu6)
    hardswish = affine_bf16 * relu6 * (1.0 / 6.0)
    out_bf16_f32 = ct.astype(ct.astype(hardswish, ct.bfloat16), ct.float32)

    h_idx = ct.arange(HP, dtype=ct.int32)
    w_idx = ct.arange(WP, dtype=ct.int32)
    h_valid = ct.reshape(h_idx < H, (HP, 1))
    w_valid = ct.reshape(w_idx < W, (1, WP))
    valid = h_valid & w_valid
    out_masked = ct.where(valid, out_bf16_f32, zero_f)
    total = ct.sum(out_masked)
    pooled = total * (1.0 / HW_F)
    ct.store(out_ptr, index=(b, c), tile=ct.reshape(ct.astype(pooled, ct.bfloat16), (1, 1)))


@oracle_impl(hardware="B200", point="3e244c1d", BLOCK_ROWS=8, BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_HW):
    running_mean, x, running_var, weight, bias, shape, stride = inputs
    out = torch.empty_strided(
        (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])),
        (int(stride[0]), int(stride[1]), int(stride[2]), int(stride[3])),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_2d = out.view(BATCH, CHANNELS)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, CHANNELS, 1),
        _bn_hardswish_spatial_mean_kernel,
        (running_mean, x, running_var, weight, bias, out_2d,
         CHANNELS, H, W, HP, WP, HW),
    )
    return out
