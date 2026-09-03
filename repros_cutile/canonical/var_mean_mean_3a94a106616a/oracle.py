"""cuTile port of var_mean_mean_3a94a106616a: MobileNetV3 BN+hardswish+spatial-mean.

Two cuTile kernels matching the Triton reference structure:
1. `_bn_stats_kernel` — per-channel mean/var/invstd + running-stat updates via ct.sum
2. `_hardswish_spatial_mean_kernel` — affine + bf16 round + hardswish + spatial
   mean via ct.sum

Shape [32, 960, 7, 7] NCHW contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-3
MOMENTUM = 0.01
RUNNING_VAR_CORRECTION = 1.0006381620931717


@ct.kernel
def _bn_stats_kernel(
    x_ptr,               # bf16 [C, E]  (permuted [C, N*HW])
    running_mean_ptr,    # f32 [C]
    running_var_ptr,     # f32 [C]
    mean_ptr,            # f32 [C]
    invstd_ptr,          # f32 [C]
    E: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    channel = ct.bid(0)
    x = ct.load(
        x_ptr,
        index=(channel, 0),
        shape=(1, BLOCK_E),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    sum_x = ct.sum(x_f)
    sum_x2 = ct.sum(x_f * x_f)
    inv_e = 1.0 / E
    mean = sum_x * inv_e
    var = sum_x2 * inv_e - mean * mean
    zero_v = ct.full((), 0.0, dtype=ct.float32)
    var = ct.where(var < zero_v, zero_v, var)
    invstd = ct.rsqrt(var + EPS)

    old_mean = ct.load(running_mean_ptr, index=(channel,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(channel,), shape=(1,))

    mean_1 = ct.reshape(mean, (1,))
    var_1 = ct.reshape(var, (1,))
    invstd_1 = ct.reshape(invstd, (1,))

    new_mean = old_mean * (1.0 - MOMENTUM) + mean_1 * MOMENTUM
    corrected_var = var_1 * RUNNING_VAR_CORRECTION
    new_var = old_var * (1.0 - MOMENTUM) + corrected_var * MOMENTUM

    ct.store(mean_ptr, index=(channel,), tile=mean_1)
    ct.store(invstd_ptr, index=(channel,), tile=invstd_1)
    ct.store(running_mean_ptr, index=(channel,), tile=new_mean)
    ct.store(running_var_ptr, index=(channel,), tile=new_var)


@ct.kernel
def _hardswish_spatial_mean_kernel(
    x_ptr,           # bf16 [N*C, HW]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    mean_out_ptr,    # bf16 [N*C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row_block_id = ct.bid(0)
    c_blocks = C // ROW_BLOCK
    c_block = row_block_id % c_blocks

    x = ct.load(
        x_ptr,
        index=(row_block_id, 0),
        shape=(ROW_BLOCK, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(ROW_BLOCK,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(ROW_BLOCK,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(ROW_BLOCK,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(ROW_BLOCK,))

    mean_col = ct.reshape(mean, (ROW_BLOCK, 1))
    invstd_col = ct.reshape(invstd, (ROW_BLOCK, 1))
    weight_col = ct.reshape(weight, (ROW_BLOCK, 1))
    bias_col = ct.reshape(bias, (ROW_BLOCK, 1))

    normalized = (x_f - mean_col) * invstd_col
    affine = normalized * weight_col + bias_col
    rounded = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    zero = ct.full((ROW_BLOCK, BLOCK_HW), 0.0, dtype=ct.float32)
    three = ct.full((ROW_BLOCK, BLOCK_HW), 3.0, dtype=ct.float32)
    six = ct.full((ROW_BLOCK, BLOCK_HW), 6.0, dtype=ct.float32)
    relu6 = rounded + three
    relu6 = ct.where(relu6 < zero, zero, relu6)
    relu6 = ct.where(relu6 > six, six, relu6)
    hswish = rounded * relu6 * (1.0 / 6.0)
    hswish_bf16 = ct.astype(hswish, ct.bfloat16)
    hswish_f = ct.astype(hswish_bf16, ct.float32)

    # Mask HW positions beyond HW so padding contributes 0 to spatial sum.
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_valid = hw_idx < HW
    hw_valid_2d = ct.reshape(hw_valid, (1, BLOCK_HW))
    hswish_masked = ct.where(hw_valid_2d, hswish_f, zero)
    reduced = ct.sum(hswish_masked, axis=1)  # (ROW_BLOCK,)
    mean_value = ct.astype(reduced * (1.0 / HW), ct.bfloat16)
    ct.store(mean_out_ptr, index=(row_block_id,), tile=mean_value)


@oracle_impl(
    hardware="B200",
    point="e8a9d13a",
    BLOCK_E=2048,
    BLOCK_HW=64,
    ROW_BLOCK=8,
)
def oracle_forward(inputs, *, BLOCK_E: int, BLOCK_HW: int, ROW_BLOCK: int):
    x, running_mean, running_var, weight, bias, _shape_param = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    total_rows = n * c
    device = x.device

    # Stats kernel wants per-channel access; permute NCHW -> [C, N*H*W].
    x_perm = x.permute(1, 0, 2, 3).contiguous().view(c, e)

    # Epilogue kernel loads spatial rows; NCHW contiguous -> [N*C, H*W] (view).
    x_flat = x.contiguous().view(total_rows, hw)

    saved_mean = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1),
        (c, 1, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    spatial_mean = torch.empty_strided(
        (n, c),
        (c, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    saved_mean_flat = saved_mean.view(c)
    invstd_flat = invstd.view(c)
    spatial_mean_flat = spatial_mean.view(total_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_stats_kernel,
        (x_perm, running_mean, running_var, saved_mean_flat, invstd_flat,
         e, BLOCK_E),
    )
    ct.launch(
        stream,
        (ct.cdiv(total_rows, ROW_BLOCK), 1, 1),
        _hardswish_spatial_mean_kernel,
        (x_flat, saved_mean_flat, invstd_flat, weight, bias,
         spatial_mean_flat, c, hw, ROW_BLOCK, BLOCK_HW),
    )

    return saved_mean, invstd, spatial_mean, running_mean, running_var
