"""cuTile port of sum_sum_50a94749c62d: GhostNet BN-backward slice+add.

One program per channel. Load [N, HW_PAD] tile (padded from HW=196).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 9.964923469387754e-06


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_backward_kernel(
    wide_arr,        # bf16 [N, INPUT_C, HW]
    residual_arr,    # bf16 [N, C, HW]
    x_arr,           # bf16 [N, C, HW] (centered_source)
    mean_arr,        # f32 [C]
    invstd_arr,      # f32 [C]
    weight_arr,      # f32 [C]
    sum_out_arr,     # f32 [C]
    weight_grad_arr, # f32 [C]
    add_out_arr,     # bf16 [N, C, HW]
    N_: ct.Constant[int],
    HW_: ct.Constant[int],
    HW_PAD: ct.Constant[int],
):
    c = ct.bid(0)

    # channels-last strided input is stored as [N, HW, C]; but here inputs
    # arg0_1 and arg1_1 have contiguous format. Let me confirm... actually
    # per Repro forward, all inputs and outputs have channels-last strides.
    # But the reference kernel indexes as wide_ptr + rows*112 + cols using 2D.

    # We'll operate on a [N, C, HW] logical view (assumes contiguous NCHW-flat).
    wide = ct.load(wide_arr, index=(0, c, 0), shape=(N_, 1, HW_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_arr, index=(0, c, 0), shape=(N_, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_arr, index=(0, c, 0), shape=(N_, 1, HW_PAD),
                padding_mode=ct.PaddingMode.ZERO)

    wide_f = ct.astype(wide, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    x_f = ct.astype(x, ct.float32)

    # add_value = (wide + residual), rounded to bf16
    add_bf16 = ct.astype(wide_f + residual_f, ct.bfloat16)
    add_f = ct.astype(add_bf16, ct.float32)

    # Mask out OOB HW positions
    hw_idx = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_idx < HW_
    hw_valid_3d = ct.reshape(hw_valid, (1, 1, HW_PAD))
    zero = ct.full(shape=(N_, 1, HW_PAD), fill_value=0.0, dtype=ct.float32)
    add_masked = ct.where(hw_valid_3d, add_f, zero)

    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_arr, index=(c,), shape=(1,))
    weight = ct.load(weight_arr, index=(c,), shape=(1,))
    mean_bc = ct.reshape(mean, (1, 1, 1))
    invstd_bc = ct.reshape(invstd, (1, 1, 1))
    weight_bc = ct.reshape(weight, (1, 1, 1))

    centered = ct.where(hw_valid_3d, x_f - mean_bc, zero)

    sum_add = ct.sum(add_masked)
    sum_prod = ct.sum(add_masked * centered)

    mean_term = sum_add * SCALE
    dot_scaled = sum_prod * SCALE
    variance_term = dot_scaled * invstd_bc * invstd_bc
    output_scale = invstd_bc * weight_bc

    grad = (add_f - centered * variance_term - mean_term) * output_scale
    grad_bf16 = ct.astype(grad, ct.bfloat16)

    # Store with masking on HW positions.
    n_idx = ct.arange(N_, dtype=ct.int32)
    c_idx_bc = ct.full(shape=(N_, 1, HW_PAD), fill_value=c, dtype=ct.int32)
    n_bc = ct.broadcast_to(ct.reshape(n_idx, (N_, 1, 1)), (N_, 1, HW_PAD))
    hw_bc = ct.broadcast_to(ct.reshape(hw_idx, (1, 1, HW_PAD)), (N_, 1, HW_PAD))
    valid_bc = ct.broadcast_to(hw_valid_3d, (N_, 1, HW_PAD))
    ct.scatter(add_out_arr, (n_bc, c_idx_bc, hw_bc), grad_bf16, mask=valid_bc)

    ct.store(sum_out_arr, index=(c,), tile=ct.reshape(sum_add, (1,)))
    ct.store(weight_grad_arr, index=(c,),
             tile=ct.reshape(sum_prod * invstd_bc, (1,)))


@oracle_impl(hardware="B200", point="fe80348b")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    n = int(arg0_1.shape[0])
    input_c = int(arg0_1.shape[1])
    c = int(arg1_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    device = arg0_1.device

    sum_out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    weight_grad = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, 1, c * w, c),
        device=device,
        dtype=torch.bfloat16,
    )

    # Inputs use channels-last strides. Convert to [N, C, HW] contiguous for our kernel.
    # The slice happens on channels 0..56 of arg0_1.
    wide_slice = arg0_1[:, :c, :, :].contiguous()
    residual_c = arg1_1.contiguous()
    x_c = arg2_1.contiguous()

    wide_3d = wide_slice.view(n, c, hw)
    residual_3d = residual_c.view(n, c, hw)
    x_3d = x_c.view(n, c, hw)
    mean_1d = arg3_1.view(c)
    invstd_1d = arg4_1.view(c)
    weight_1d = arg5_1.view(c)

    # For the output — need to write to channels-last output.
    add_out_temp = torch.empty((n, c, hw), device=device, dtype=torch.bfloat16)

    HW_PAD = _next_pow2(hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_backward_kernel,
        (wide_3d, residual_3d, x_3d,
         mean_1d, invstd_1d, weight_1d,
         sum_out, weight_grad, add_out_temp,
         n, hw, HW_PAD),
    )

    # Copy temp NCHW back to channels-last output
    add_out.copy_(add_out_temp.view(n, c, h, w))
    return sum_out, weight_grad, add_out
