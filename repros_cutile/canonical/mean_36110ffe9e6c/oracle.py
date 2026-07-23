"""cuTile port of mean_36110ffe9e6c: BN + ReLU + 7x7 spatial mean (MnasNet/ShuffleNet)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW_SIZE = 49
EPS = 1.0e-5


@ct.kernel
def _bn_relu_spatial_mean_kernel(
    x_ptr,        # bf16 [N, C, H, W] = flat [N*C, HW]
    mean_ptr,     # bf16 [C]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [N, C]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    # Load channel-affine parameters (scalar per row)
    mean_v = ct.load(mean_ptr, index=(c,), shape=(1,))
    var_v = ct.load(var_ptr, index=(c,), shape=(1,))
    weight_v = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_v = ct.load(bias_ptr, index=(c,), shape=(1,))

    mean_f = ct.astype(mean_v, ct.float32)
    var_f = ct.astype(var_v, ct.float32)
    weight_f = ct.astype(weight_v, ct.float32)
    bias_f = ct.astype(bias_v, ct.float32)

    invstd = ct.rsqrt(var_f + EPS)
    scale = invstd * weight_f  # scalar
    shift = bias_f - mean_f * scale  # scalar

    # Load HW block (padded with zeros beyond HW=49)
    x = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    # affine = x * scale + shift (broadcasted per-element)
    affine = x_f * scale + shift
    affine_bf = ct.astype(affine, ct.bfloat16)
    # ReLU
    zero_bf = ct.astype(ct.zeros(shape=(1, 1, BLOCK_HW), dtype=ct.float32), ct.bfloat16)
    relu = ct.where(affine_bf < zero_bf, zero_bf, affine_bf)
    # Zero-out OOB positions (padding was already zero, but shift adds nonzero)
    # Build mask via arange < HW
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_mask = hw_idx < HW_C
    hw_mask_3d = ct.reshape(hw_mask, (1, 1, BLOCK_HW))
    relu_f = ct.astype(relu, ct.float32)
    relu_masked = ct.where(hw_mask_3d, relu_f, 0.0)
    total = ct.sum(relu_masked) * (1.0 / HW_C)
    result_bf = ct.astype(ct.full(shape=(1,), fill_value=total, dtype=ct.float32), ct.bfloat16)
    # out shape is (N, C); store one element at (n, c)
    ct.store(out_ptr, index=(n, c), tile=ct.reshape(result_bf, (1, 1)))


@oracle_impl(hardware="B200", point="8444bfb4", BLOCK_HW=64)
@oracle_impl(hardware="B200", point="2adc7e85", BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_HW: int):
    mean, x, var, weight, bias = inputs
    n_size = int(x.shape[0])
    channels = int(x.shape[1])
    # Flatten spatial dims: view as (n_size, channels, H*W)
    x_flat = x.view(n_size, channels, HW_SIZE)
    out = torch.empty_strided(
        (n_size, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_size, channels, 1),
        _bn_relu_spatial_mean_kernel,
        (x_flat, mean, var, weight, bias, out, channels, HW_SIZE, BLOCK_HW),
    )
    return out
