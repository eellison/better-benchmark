"""cuTile port of mean_bae5a06d2559: MobileNetV2 BN+ReLU6+spatial_mean.

For each (batch, channel) row: BN affine over HW=49 spatial elements, ReLU6
clamped to [0,6], spatial mean over 49, final bf16 output shape (N, C).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW_POW2 = 64  # next pow2 >= 49


@ct.kernel
def _bn_relu6_spatial_mean_kernel(
    x_ptr,         # bf16 (N, C, HW_POW2) — padded with zero beyond HW
    mean_ptr,      # bf16 (C,)
    var_ptr,       # bf16 (C,)
    weight_ptr,    # bf16 (C,)
    bias_ptr,      # bf16 (C,)
    out_ptr,       # bf16 (N, C)
    HW_C: ct.Constant[int],
    HW_POW2_C: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    # Load the spatial slice for (n, c) - shape (1, 1, HW_POW2)
    x = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, HW_POW2_C))
    x_f = ct.astype(x, ct.float32)

    mean_bf16 = ct.load(mean_ptr, index=(c,), shape=(1,))
    var_bf16 = ct.load(var_ptr, index=(c,), shape=(1,))
    weight_bf16 = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_bf16 = ct.load(bias_ptr, index=(c,), shape=(1,))
    mean_f = ct.astype(mean_bf16, ct.float32)
    var_f = ct.astype(var_bf16, ct.float32)
    weight_f = ct.astype(weight_bf16, ct.float32)
    bias_f = ct.astype(bias_bf16, ct.float32)

    # BN affine: (x - mean) / sqrt(var + eps) * weight + bias
    denom = ct.sqrt(var_f + 1.0e-5)
    invstd = 1.0 / denom
    normalized = (x_f - mean_f) * invstd
    affine = normalized * weight_f + bias_f
    # bf16 round-trip
    rounded = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    zero_tile = ct.zeros(shape=(1, 1, HW_POW2_C), dtype=ct.float32)
    six_tile = ct.full(shape=(1, 1, HW_POW2_C), fill_value=6.0, dtype=ct.float32)
    relu6 = ct.where(rounded < zero_tile, zero_tile, rounded)
    relu6 = ct.where(relu6 > six_tile, six_tile, relu6)
    # bf16 round-trip
    relu6_bf16 = ct.astype(ct.astype(relu6, ct.bfloat16), ct.float32)
    # Mask OOB positions with zero for sum
    idx = ct.arange(HW_POW2_C, dtype=ct.int32)
    idx_3d = ct.reshape(idx, (1, 1, HW_POW2_C))
    mask = idx_3d < HW_C
    masked = ct.where(mask, relu6_bf16, ct.zeros(shape=(1, 1, HW_POW2_C), dtype=ct.float32))
    spatial_sum = ct.sum(masked)
    spatial_mean = spatial_sum * (1.0 / 49.0)
    out_bf16 = ct.astype(spatial_mean, ct.bfloat16)
    ct.store(out_ptr, index=(n, c), tile=out_bf16)


@oracle_impl(hardware="B200", point="c92cf8b8", BLOCK_ROWS=32, BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int):
    running_mean, x, running_var, weight, bias, view_shape = inputs
    N = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    HW = H * W  # 49

    # Pad x to (N, C, HW_POW2) with zeros in the tail
    x_flat = x.view(N, C, HW)
    x_padded = torch.zeros((N, C, HW_POW2), device=x.device, dtype=torch.bfloat16)
    x_padded[:, :, :HW] = x_flat

    out = torch.empty_strided(
        tuple(int(dim) for dim in view_shape),
        (C, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C, 1),
        _bn_relu6_spatial_mean_kernel,
        (x_padded, running_mean, running_var, weight, bias, out, HW, HW_POW2),
    )
    return out
