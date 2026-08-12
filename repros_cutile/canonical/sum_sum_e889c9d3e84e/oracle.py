"""cuTile port of sum_sum_e889c9d3e84e: DenseNet BN-backward per-channel reduction + residual add epilogue.

One program per channel. Loads the [N, HW] slice for that channel, reduces to
sum_where and sum_dot, then computes the affine BN-backward output and adds the
sliced residual.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 120
H = 4
W = 4
HW = H * W
INV_COUNT = 0.00048828125


@ct.kernel
def _bn_backward_kernel(
    residual_arr,     # bf16 [N, 136, HW]  (residual, sliced offset by 16 channels)
    mask_input_arr,   # bf16 [N, C, HW]
    full_arr,         # bf16 scalar
    source_arr,       # bf16 [N, C, HW]
    centered_base_arr, # bf16 [N, C, HW]
    mean_arr,         # f32 [C]
    invstd_arr,       # f32 [C]
    affine_weight_arr, # f32 [C]
    sum_where_arr,    # f32 [C]
    weight_grad_arr,  # f32 [C]
    out_arr,          # bf16 [N, C, HW]
    N_: ct.Constant[int],
    HW_: ct.Constant[int],
    N_HW: ct.Constant[int],   # BLOCK N*HW
):
    c = ct.bid(0)

    # Load [N, HW] slice — 128*16 = 2048 elements, power-of-2
    mask_input = ct.load(mask_input_arr, index=(0, c, 0), shape=(N_, 1, HW_))
    source = ct.load(source_arr, index=(0, c, 0), shape=(N_, 1, HW_))
    centered_base = ct.load(centered_base_arr, index=(0, c, 0), shape=(N_, 1, HW_))
    full_scalar = ct.load(full_arr, index=(0,), shape=(1,))

    # Broadcast full_scalar to the tile shape
    mask_input_f = ct.astype(mask_input, ct.float32)
    source_f = ct.astype(source, ct.float32)
    centered_base_f = ct.astype(centered_base, ct.float32)
    full_f = ct.astype(full_scalar, ct.float32)

    # Scatter full's value across all elements
    full_broadcast = ct.reshape(full_f, (1, 1, 1))

    # where_value = where(mask_input <= 0, full_value, source)
    where_value = ct.where(mask_input_f <= 0.0, full_broadcast, source_f)

    # Per-channel scalars
    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_arr, index=(c,), shape=(1,))
    affine_weight = ct.load(affine_weight_arr, index=(c,), shape=(1,))
    mean_f = ct.astype(mean, ct.float32)
    invstd_f = ct.astype(invstd, ct.float32)
    affine_weight_f = ct.astype(affine_weight, ct.float32)

    # centered = centered_base - mean
    mean_3d = ct.reshape(mean_f, (1, 1, 1))
    centered = centered_base_f - mean_3d

    sum_where = ct.sum(where_value)
    sum_centered = ct.sum(where_value * centered)

    mean_term = sum_where * INV_COUNT
    scaled_centered_sum = sum_centered * INV_COUNT
    variance_term = scaled_centered_sum * invstd_f * invstd_f
    post_scale = invstd_f * affine_weight_f

    grad = (where_value - centered * variance_term - mean_term) * post_scale
    grad_bf16 = ct.astype(grad, ct.bfloat16)

    # Residual: [N, 136, HW] with slice offset 16 channels — so channel index is c+16
    residual = ct.load(residual_arr, index=(0, c + 16, 0), shape=(N_, 1, HW_))
    residual_f = ct.astype(residual, ct.float32)
    out_f = residual_f + ct.astype(grad_bf16, ct.float32)
    out_bf16 = ct.astype(out_f, ct.bfloat16)

    ct.store(out_arr, index=(0, c, 0), tile=out_bf16)
    ct.store(sum_where_arr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    weight_grad = ct.astype(sum_centered * invstd_f, ct.float32)
    ct.store(weight_grad_arr, index=(c,), tile=ct.reshape(weight_grad, (1,)))


@oracle_impl(hardware="B200", point="4055c4c5")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs

    device = arg0_1.device
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    weight_grad = torch.empty((C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Views into 3D shape [N, C_or_136, HW]
    residual_3d = arg0_1.view(N, 136, HW)
    mask_input_3d = arg1_1.view(N, C, HW)
    source_3d = arg3_1.view(N, C, HW)
    centered_base_3d = arg4_1.view(N, C, HW)
    mean_1d = arg5_1.view(C)
    invstd_1d = arg6_1.view(C)
    affine_weight_1d = arg7_1.view(C)
    out_3d = out_bf16.view(N, C, HW)
    # arg2_1 is a scalar bf16 tensor
    full_1d = arg2_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_backward_kernel,
        (residual_3d, mask_input_3d, full_1d, source_3d, centered_base_3d,
         mean_1d, invstd_1d, affine_weight_1d, sum_where, weight_grad, out_3d,
         N, HW, N * HW),
    )
    return sum_where, weight_grad, out_bf16, out_bf16[:, :16, :, :]
