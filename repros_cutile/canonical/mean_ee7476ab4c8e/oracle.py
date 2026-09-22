"""cuTile port of mean_ee7476ab4c8e: DenseNet BN-inference + ReLU + 7x7 mean.

Per-(batch, channel) row kernel: for each (batch, channel), load the 49
activations, apply channel-only BN algebra, ReLU, spatial mean, output bf16.
Launch grid is (BATCH, CHANNELS).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
CHANNELS = 1024
H = 7
W = 7
HW = H * W  # 49
BLOCK_HW = 64  # padded to next pow-of-2


@ct.kernel
def _bn_relu_spatial_mean_kernel(
    x_ptr,       # bf16 [BATCH, CHANNELS, HW]
    mean_ptr,    # bf16 [CHANNELS]
    var_ptr,     # bf16 [CHANNELS]
    weight_ptr,  # bf16 [CHANNELS]
    bias_ptr,    # bf16 [CHANNELS]
    out_ptr,     # bf16 [BATCH, CHANNELS]
):
    b = ct.bid(0)
    c = ct.bid(1)

    # Load 49 activations padded to 64.
    x = ct.load(
        x_ptr,
        index=(b, c, 0),
        shape=(1, 1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    channel_mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    channel_var = ct.load(var_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))

    mean_f = ct.astype(channel_mean, ct.float32)
    var_f = ct.astype(channel_var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    invstd = ct.rsqrt(var_f + 1.0e-5)
    # x_f is (1, 1, BLOCK_HW). Reshape mean_f, invstd etc. broadcast against it.
    mean_b = ct.reshape(mean_f, (1, 1, 1))
    invstd_b = ct.reshape(invstd, (1, 1, 1))
    weight_b = ct.reshape(weight_f, (1, 1, 1))
    bias_b = ct.reshape(bias_f, (1, 1, 1))

    normed = (x_f - mean_b) * invstd_b
    affine = normed * weight_b + bias_b
    y_bf16 = ct.astype(affine, ct.bfloat16)
    y_f32 = ct.astype(y_bf16, ct.float32)
    # ReLU (with NaN preservation): if NaN keep NaN, else max(y, 0).
    zero = ct.full(shape=(1, 1, BLOCK_HW), fill_value=0.0, dtype=ct.float32)
    relu = ct.where(y_f32 > zero, y_f32, zero)
    is_nan = y_f32 != y_f32
    relu = ct.where(is_nan, y_f32, relu)

    # Mask out padded lanes (indices >= HW) before summing.
    idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    idx_3d = ct.reshape(idx, (1, 1, BLOCK_HW))
    hw_mask = idx_3d < HW
    relu_masked = ct.where(hw_mask, relu, zero)
    total = ct.sum(relu_masked)
    mean_val = total * (1.0 / HW)
    mean_bf = ct.astype(mean_val, ct.bfloat16)
    # Store as tile of shape (1,)
    ct.store(out_ptr, index=(b, c), tile=ct.reshape(mean_bf, (1, 1)))


@oracle_impl(hardware="B200", point="2adc7e85")
def oracle_forward(inputs):
    channel_mean, x, var, weight, bias, _shape = inputs
    # x: bf16[64, 1024, 7, 7] contiguous. Reshape to [64, 1024, 49].
    x_3d = x.reshape(BATCH, CHANNELS, HW)
    out = torch.empty_strided(
        (BATCH, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, CHANNELS, 1),
        _bn_relu_spatial_mean_kernel,
        (x_3d, channel_mean, var, weight, bias, out),
    )
    return out
