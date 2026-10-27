"""cuTile port of sum_sum_8ddadf2c06d3: DenseNet BN-backward + slice output.

One program per channel. Loads the [N*HW] flat vector for that channel
(padded to power-of-2), reduces to sum_where and sum_dot, then computes
the affine BN-backward output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 1024
H = 14
W = 14
HW = H * W
R = N * HW
SCALE = 0.0012755102040816326
SLICE_START = 992


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_backward_kernel(
    mask_input_arr,   # bf16 [N*HW*C] flattened as [N, C, HW]
    fill_arr,         # bf16 scalar
    source_arr,       # bf16 [N, C, HW]
    centered_source_arr, # bf16 [N, C, HW]
    mean_arr,         # f32 [C]
    invstd_arr,       # f32 [C]
    weight_arr,       # f32 [C]
    sum_out_arr,      # f32 [C]
    scale_grad_arr,   # f32 [C]
    dense_out_arr,    # bf16 [N, C, HW]
    N_: ct.Constant[int],
    HW_: ct.Constant[int],
    HW_PAD: ct.Constant[int],
):
    c = ct.bid(0)

    # [N, 1, HW_PAD] tile — N*HW_PAD = 4*256 = 1024. OOB HW is zero-padded.
    mask_input = ct.load(
        mask_input_arr, index=(0, c, 0), shape=(N_, 1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    source = ct.load(
        source_arr, index=(0, c, 0), shape=(N_, 1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered_source = ct.load(
        centered_source_arr, index=(0, c, 0), shape=(N_, 1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill_scalar = ct.load(fill_arr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    fill_broadcast = ct.reshape(fill_f, (1, 1, 1))

    mask_input_f = ct.astype(mask_input, ct.float32)
    source_f = ct.astype(source, ct.float32)
    centered_source_f = ct.astype(centered_source, ct.float32)

    where_bf16 = ct.where(mask_input_f <= 0.0, fill_broadcast, source_f)
    # OOB positions in HW should not contribute — mask them off
    hw_idx = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_idx < HW_
    hw_valid_3d = ct.reshape(hw_valid, (1, 1, HW_PAD))
    zero_f = ct.full(shape=(N_, 1, HW_PAD), fill_value=0.0, dtype=ct.float32)
    where_f = ct.where(hw_valid_3d, where_bf16, zero_f)

    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_arr, index=(c,), shape=(1,))
    weight = ct.load(weight_arr, index=(c,), shape=(1,))
    mean_f = ct.astype(mean, ct.float32)
    invstd_f = ct.astype(invstd, ct.float32)
    weight_f = ct.astype(weight, ct.float32)

    mean_3d = ct.reshape(mean_f, (1, 1, 1))
    centered = ct.where(hw_valid_3d, centered_source_f - mean_3d, zero_f)

    sum_where = ct.sum(where_f)
    sum_centered = ct.sum(where_f * centered)

    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    variance_term = dot_scaled * invstd_f * invstd_f
    output_scale = invstd_f * weight_f

    grad = (where_f - centered * variance_term - mean_term) * output_scale
    grad_bf16 = ct.astype(grad, ct.bfloat16)

    # Scatter store to valid HW positions only. We need indices for N, C, HW.
    n_idx = ct.arange(N_, dtype=ct.int32)
    c_idx = ct.full(shape=(N_, 1, HW_PAD), fill_value=c, dtype=ct.int32)
    n_bc = ct.broadcast_to(ct.reshape(n_idx, (N_, 1, 1)), (N_, 1, HW_PAD))
    hw_bc = ct.broadcast_to(ct.reshape(hw_idx, (1, 1, HW_PAD)), (N_, 1, HW_PAD))
    valid = ct.broadcast_to(hw_valid_3d, (N_, 1, HW_PAD))
    ct.scatter(dense_out_arr, (n_bc, c_idx, hw_bc), grad_bf16, mask=valid)

    ct.store(sum_out_arr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(scale_grad_arr, index=(c,), tile=ct.reshape(sum_centered * invstd_f, (1,)))


@oracle_impl(hardware="B200", point="470762bc")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg0_1.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Views into 3D shape [N, C, HW]
    mask_input_3d = arg0_1.view(N, C, HW)
    source_3d = arg2_1.view(N, C, HW)
    centered_3d = arg3_1.view(N, C, HW)
    mean_1d = arg4_1.view(C)
    invstd_1d = arg5_1.view(C)
    weight_1d = arg6_1.view(C)
    dense_3d = dense_out.view(N, C, HW)
    fill_1d = arg1_1.view(1)

    HW_PAD = _next_pow2(HW)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_backward_kernel,
        (mask_input_3d, fill_1d, source_3d, centered_3d,
         mean_1d, invstd_1d, weight_1d, sum_out, scale_grad, dense_3d,
         N, HW, HW_PAD),
    )
    return sum_out, scale_grad, dense_out, dense_out[:, SLICE_START:C, :, :]
