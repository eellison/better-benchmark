"""cuTile port of var_mean_1f6788bc3116: DenseNet 8-input concat + BN-train + ReLU.

The 8-input torch.cat and running-stat copy_ epilogue are performed with
torch on the Python side; a single cuTile channel kernel computes the
per-channel var_mean and materializes the affine ReLU output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.005128205128205


@ct.kernel
def _bn_train_channel_kernel(
    x_ptr,             # bf16 (channels, elements_per_channel)
    weight_ptr,        # f32 (channels,)
    bias_ptr,          # f32 (channels,)
    invstd_out_ptr,    # f32 (channels,)
    mean_out_ptr,      # f32 (channels,)
    var_out_ptr,       # f32 (channels,)
    relu_out_ptr,      # bf16 (channels, elements_per_channel)
    ELEMENTS_PER_CHANNEL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    channel = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(channel, 0), shape=(1, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_bf, ct.float32)

    cols = ct.arange(BLOCK_K, dtype=ct.int32)
    col_valid = ct.reshape(cols < ELEMENTS_PER_CHANNEL, (1, BLOCK_K))
    zero_f = ct.zeros((1, BLOCK_K), dtype=ct.float32)
    x_masked = ct.where(col_valid, x_f, zero_f)

    sum_x = ct.sum(x_masked)
    sum_x2 = ct.sum(x_masked * x_masked)
    mean = sum_x * (1.0 / ELEMENTS_PER_CHANNEL)
    var = sum_x2 * (1.0 / ELEMENTS_PER_CHANNEL) - mean * mean
    zero_scalar = ct.zeros((1,), dtype=ct.float32)
    var_1d = ct.reshape(var, (1,))
    var_clamped_1d = ct.where(var_1d > zero_scalar, var_1d, zero_scalar)
    invstd_1d = ct.rsqrt(var_clamped_1d + EPS)
    mean_1d = ct.reshape(mean, (1,))

    ct.store(invstd_out_ptr, index=(channel,), tile=invstd_1d)
    ct.store(mean_out_ptr, index=(channel,), tile=mean_1d)
    ct.store(var_out_ptr, index=(channel,), tile=var_clamped_1d)

    weight = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias = ct.load(bias_ptr, index=(channel,), shape=(1,))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))
    invstd_2d = ct.reshape(invstd_1d, (1, 1))
    mean_2d = ct.reshape(mean_1d, (1, 1))

    y = (x_f - mean_2d) * invstd_2d
    y = y * weight_2d + bias_2d
    y_bf16 = ct.astype(y, ct.bfloat16)
    zero_bf = ct.zeros((1, BLOCK_K), dtype=ct.bfloat16)
    is_nan = y_bf16 != y_bf16
    relu = ct.where(is_nan, y_bf16, ct.where(y_bf16 > zero_bf, y_bf16, zero_bf))
    ct.store(relu_out_ptr, index=(channel, 0), tile=relu)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _run(inputs, *, C0: int, H: int, W: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1, arg11_1,
    ) = inputs
    n = int(arg0_1.shape[0])
    branch_c = 32
    branches = 7
    channels = C0 + branches * branch_c
    hw = H * W
    elements_per_channel = n * hw

    cat = torch.cat([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1], dim=1)
    cat_reshaped = cat.permute(1, 0, 2, 3).contiguous().view(channels, elements_per_channel)

    invstd = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    mean_out = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    var_out = torch.empty((channels,), device=arg0_1.device, dtype=torch.float32)
    relu_reshaped = torch.empty((channels, elements_per_channel),
                                  device=arg0_1.device, dtype=torch.bfloat16)

    BLOCK_K = _next_pow2(elements_per_channel)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, 1, 1),
        _bn_train_channel_kernel,
        (cat_reshaped, arg10_1, arg11_1,
         invstd, mean_out, var_out, relu_reshaped,
         elements_per_channel, BLOCK_K),
    )

    # Update running stats via copy_.
    new_running_mean = arg8_1 * 0.9 + mean_out * 0.1
    new_running_var = arg9_1 * 0.9 + var_out * RUNNING_VAR_CORRECTION * 0.1
    torch.ops.aten.copy_.default(arg8_1, new_running_mean)
    torch.ops.aten.copy_.default(arg9_1, new_running_var)

    # Reshape relu back to (n, channels, H, W).
    relu = relu_reshaped.view(channels, n, H, W).permute(1, 0, 2, 3).contiguous()

    mean_saved = mean_out.view(1, channels, 1, 1)
    return cat, invstd, relu, mean_saved, arg8_1, arg9_1


@oracle_impl(hardware="B200", point="7b2f839c", C0=512, H=7, W=7)
@oracle_impl(hardware="B200", point="49e8ad15", C0=256, H=14, W=14)
@oracle_impl(hardware="B200", point="a4825250", C0=128, H=28, W=28)
def oracle_forward(inputs, *, C0: int, H: int, W: int):
    return _run(inputs, C0=C0, H=H, W=W)
