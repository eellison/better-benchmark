"""cuTile port of mean_9fffeb5504ef (SCHEDULER_FUSION): MobileNetV3 channels-last
BN-inference affine + ReLU + spatial mean, per (batch, channel) row.

For row `r` = (n, c):
  y = (x - mean_c) * (1/sqrt(var_c + eps)) * weight_c + bias_c
  relu = max(y, 0), stored to relu_out
  mean_out = mean(relu, over hw)  ← reduced over spatial 784 pixels
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_relu_mean_kernel(
    mean_ptr,     # bf16 [C]
    x_ptr,        # bf16 [N*C, HW] (row-major)
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    relu_out_ptr, # bf16 [N*C, HW]
    mean_out_ptr, # bf16 [N*C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)
    c = row % C

    # Load per-channel scalars using shape=(1,).
    mean_c = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var_c = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    weight_c = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias_c = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)
    invstd = ct.rsqrt(var_c + EPS)

    # Load activation row [1, HW]. Pad zero for OOB (HW <= BLOCK_HW).
    x = ct.load(
        x_ptr,
        index=(row, 0),
        shape=(1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    y = (x_f - mean_c) * invstd * weight_c + bias_c
    zero_f = ct.full((1, BLOCK_HW), 0.0, dtype=ct.float32)
    # Preserve NaN: max(0, NaN) -> NaN, so use where.
    is_nan = y != y
    relu = ct.where(is_nan, y, ct.maximum(y, zero_f))
    relu_bf16 = ct.astype(relu, ct.bfloat16)
    ct.store(relu_out_ptr, index=(row, 0), tile=relu_bf16)

    # Compute mean over the valid HW window: sum(relu * (col_mask ? 1 : 0)) / HW
    cols = ct.arange(BLOCK_HW, dtype=ct.int32)
    col_valid = cols < HW
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_HW))
    relu_f = ct.astype(relu_bf16, ct.float32)
    masked = ct.where(col_valid_2d, relu_f, zero_f)
    total = ct.sum(masked)
    reduced = ct.astype(total * (1.0 / HW), ct.bfloat16)
    # Store scalar into [1, 1] position at row.
    reduced_1 = ct.reshape(reduced, (1,))
    ct.store(mean_out_ptr, index=(row,), tile=reduced_1)


@oracle_impl(hardware="B200", point="37cf4567", BLOCK_HW=1024)
@oracle_impl(hardware="B200", point="bddd3dfb", BLOCK_HW=1024)
def oracle_forward(inputs, *, BLOCK_HW):
    mean, x, var, weight, bias = inputs
    batch, channels, height, width = x.shape
    hw = height * width  # 784

    # x is channels-last strided; make a contiguous [N*C, HW] view.
    x_ncl = x.reshape(batch * channels, hw).contiguous() if not x.is_contiguous(
        memory_format=torch.contiguous_format) else x.view(batch * channels, hw)
    # But we need x reordered. x is channels-last (strides [C*H*W, 1, W*C, C]).
    # x.permute to NCHW contiguous first.
    x_nchw = x.contiguous()
    x_flat = x_nchw.view(batch * channels, hw)

    relu_out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    # Materialize relu in NCHW contiguous then permute back to channels-last.
    relu_temp = torch.empty(
        (batch * channels, hw),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_flat = mean_out.view(batch * channels)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch * channels, 1, 1),
        _bn_relu_mean_kernel,
        (mean, x_flat, var, weight, bias, relu_temp, mean_flat, channels, hw, BLOCK_HW),
    )

    # Now copy relu_temp (NCHW contiguous) into relu_out (NHWC-strided).
    relu_nchw = relu_temp.view(batch, channels, height, width)
    relu_out.copy_(relu_nchw)
    return relu_out, mean_out
