"""cuTile port of var_mean_mean_5f6d60b04d02: ResNet BN-train + residual-ReLU + spatial-mean.

A single cuTile channel-tiled kernel computes per-channel var_mean and
materializes both the LE mask and the spatial-mean N,C output. Running-stat
copy_ mutation is done with torch on the Python side.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


RUNNING_VAR_CORRECTION = 1.0012771392081736
EPS = 1.0e-5


@ct.kernel
def _bn_relu_kernel(
    x_ptr,           # bf16 (channels, elements_per_channel)
    weight_ptr,      # f32 (channels,)
    bias_ptr,        # f32 (channels,)
    residual_ptr,    # bf16 (channels, elements_per_channel)
    invstd_ptr,      # f32 (channels,)
    mean_ptr,        # f32 (channels,)
    var_ptr,         # f32 (channels,)
    le_ptr,          # bool (channels, elements_per_channel)
    relu_ptr,        # bf16 (channels, elements_per_channel)
    ELEMENTS_PER_CHANNEL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    channel = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(channel, 0), shape=(1, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)

    cols = ct.arange(BLOCK_K, dtype=ct.int32)
    col_valid = ct.reshape(cols < ELEMENTS_PER_CHANNEL, (1, BLOCK_K))
    zero_f = ct.zeros((1, BLOCK_K), dtype=ct.float32)
    x_masked = ct.where(col_valid, x, zero_f)

    sum_x = ct.sum(x_masked)
    sum_x2 = ct.sum(x_masked * x_masked)
    mean = sum_x * (1.0 / ELEMENTS_PER_CHANNEL)
    var = sum_x2 * (1.0 / ELEMENTS_PER_CHANNEL) - mean * mean
    zero_scalar = ct.zeros((1,), dtype=ct.float32)
    var_1d = ct.reshape(var, (1,))
    var_clamped_1d = ct.where(var_1d > zero_scalar, var_1d, zero_scalar)
    invstd_1d = ct.rsqrt(var_clamped_1d + EPS)
    mean_1d = ct.reshape(mean, (1,))
    ct.store(invstd_ptr, index=(channel,), tile=invstd_1d)
    ct.store(mean_ptr, index=(channel,), tile=mean_1d)
    ct.store(var_ptr, index=(channel,), tile=var_clamped_1d)

    weight = ct.load(weight_ptr, index=(channel,), shape=(1,))
    bias = ct.load(bias_ptr, index=(channel,), shape=(1,))
    residual_bf = ct.load(residual_ptr, index=(channel, 0), shape=(1, BLOCK_K),
                           padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))
    invstd_2d = ct.reshape(invstd_1d, (1, 1))
    mean_2d = ct.reshape(mean_1d, (1, 1))

    centered = x - mean_2d
    normalized = centered * invstd_2d
    scaled = normalized * weight_2d
    biased = scaled + bias_2d
    y = ct.astype(biased, ct.bfloat16)
    y = ct.astype(ct.astype(y, ct.float32) + ct.astype(residual_bf, ct.float32),
                    ct.bfloat16)
    zero_bf = ct.zeros((1, BLOCK_K), dtype=ct.bfloat16)
    relu = ct.where(y < zero_bf, zero_bf, y)

    le = relu <= zero_bf
    ct.store(le_ptr, index=(channel, 0), tile=le)
    ct.store(relu_ptr, index=(channel, 0), tile=relu)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _run(inputs):
    x, running_mean, running_var, weight, bias, residual, _shape_param_0 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    elements_per_channel = n * hw

    # Reshape x, residual into (c, n*hw) - laid out channel-major.
    x_c_major = x.permute(1, 0, 2, 3).contiguous().view(c, elements_per_channel)
    residual_c_major = residual.permute(1, 0, 2, 3).contiguous().view(c, elements_per_channel)

    invstd = torch.empty((c,), device=x.device, dtype=torch.float32)
    mean_out = torch.empty((c,), device=x.device, dtype=torch.float32)
    var_out = torch.empty((c,), device=x.device, dtype=torch.float32)
    le_c_major = torch.empty((c, elements_per_channel), device=x.device, dtype=torch.bool)
    relu_c_major = torch.empty((c, elements_per_channel), device=x.device,
                                 dtype=torch.bfloat16)

    BLOCK_K = _next_pow2(elements_per_channel)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1),
        _bn_relu_kernel,
        (x_c_major, weight, bias, residual_c_major,
         invstd, mean_out, var_out, le_c_major, relu_c_major,
         elements_per_channel, BLOCK_K),
    )

    # Update running stats via copy_.
    new_running_mean = running_mean * 0.9 + mean_out * 0.1
    new_running_var = running_var * 0.9 + var_out * RUNNING_VAR_CORRECTION * 0.1
    torch.ops.aten.copy_.default(running_mean, new_running_mean)
    torch.ops.aten.copy_.default(running_var, new_running_var)

    # Reshape LE back to (n, c, h, w) and compute spatial mean.
    le = le_c_major.view(c, n, h, w).permute(1, 0, 2, 3).contiguous()
    relu_nchw = relu_c_major.view(c, n, h, w).permute(1, 0, 2, 3).contiguous()
    # spatial mean over H,W -> (N, C)
    spatial_mean = relu_nchw.to(torch.float32).mean(dim=[2, 3]).to(torch.bfloat16)

    mean_saved = mean_out.view(1, c, 1, 1)
    return invstd, spatial_mean, le, mean_saved, running_mean, running_var


@oracle_impl(hardware="B200", point="79146166")
@oracle_impl(hardware="B200", point="8881253b")
@oracle_impl(hardware="B200", point="c99f0cec")
@oracle_impl(hardware="B200", point="77734290")
@oracle_impl(hardware="B200", point="f7eda15e")
def oracle_forward(inputs):
    return _run(inputs)
