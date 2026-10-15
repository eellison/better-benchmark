"""cuTile port of var_mean_a716d6413817: DenseNet 7-branch cat + BN + ReLU.

Strategy:
* torch: build the concatenated tensor via aten.cat (fine at any shape).
* cuTile kernel per channel: compute mean/var over N*H*W elements, emit
  saved_mean/invstd, and store the BN affine ReLU output for that channel.
* torch after: update running_mean / running_var via copy_ (population var
  correction factor from the repro), squeeze mean/rsqrt into the exact
  strided outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
BRANCHES = 6  # 6 extra branches beyond arg0


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@ct.kernel
def _bn_channel_kernel(
    cat_ptr,       # bf16 [C, N*H*W]  (channel-major view)
    weight_ptr,    # f32 [C]
    bias_ptr,      # f32 [C]
    invstd_ptr,    # f32 [C]
    saved_mean_ptr,  # f32 [C]
    var_ptr,       # f32 [C]  (biased/population variance)
    relu_ptr,      # bf16 [C, N*H*W]  (same channel-major view)
    ELEMS: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    EPS_: ct.Constant[float],
):
    c = ct.bid(0)
    row = ct.load(cat_ptr, index=(c, 0), shape=(1, BLOCK_HW),
                  padding_mode=ct.PaddingMode.ZERO)
    row_f = ct.astype(row, ct.float32)

    cols = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid = ct.reshape(cols < ELEMS, (1, BLOCK_HW))
    zero_f = ct.zeros((1, BLOCK_HW), dtype=ct.float32)
    row_masked = ct.where(valid, row_f, zero_f)

    inv_n = 1.0 / ELEMS
    sum_x = ct.sum(row_masked, axis=1, keepdims=True)
    mean = sum_x * inv_n
    sum_x2 = ct.sum(row_masked * row_masked, axis=1, keepdims=True)
    variance = sum_x2 * inv_n - mean * mean
    variance = ct.where(variance > 0.0, variance, 0.0)
    invstd = ct.rsqrt(variance + EPS_)

    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    weight_2d = ct.reshape(weight, (1, 1))
    bias_2d = ct.reshape(bias, (1, 1))
    centered = row_f - mean
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((1, BLOCK_HW), dtype=ct.bfloat16)
    relu = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)

    ct.store(relu_ptr, index=(c, 0), tile=relu)

    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))
    ct.store(saved_mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(var_ptr, index=(c,), tile=ct.reshape(variance, (1,)))


@oracle_impl(hardware="B200", point="d1878f47")
@oracle_impl(hardware="B200", point="52578f8a")
@oracle_impl(hardware="B200", point="4d48d8e3")
@oracle_impl(hardware="B200", point="6d5d4f19")
def oracle_forward(inputs):
    (x0, x1, x2, x3, x4, x5, x6,
     running_mean, running_var, weight, bias) = inputs
    device = x0.device

    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    branch_c = int(x1.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    channels = c0 + BRANCHES * branch_c
    hw_size = h * w
    elements_per_channel = n * hw_size

    # 1) Materialize the concatenated tensor via aten.cat.
    cat = torch.cat([x0, x1, x2, x3, x4, x5, x6], 1).contiguous()

    # 2) Rearrange cat to (C, N*H*W) via permute [1, 0, 2, 3] -> reshape.
    cat_channel_major = (
        cat.permute(1, 0, 2, 3)
           .contiguous()
           .view(channels, elements_per_channel)
    )

    # Output buffers.
    invstd_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    saved_mean_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    variance_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    relu_channel_major = torch.empty(
        (channels, elements_per_channel), device=device, dtype=torch.bfloat16,
    )

    block_hw = _next_pow2(elements_per_channel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, 1, 1),
        _bn_channel_kernel,
        (cat_channel_major, weight, bias, invstd_1d, saved_mean_1d,
         variance_1d, relu_channel_major, elements_per_channel, block_hw, EPS),
    )

    # 3) Reshape relu back to (N, C, H, W) contiguous.
    relu = (relu_channel_major.view(channels, n, h, w)
                              .permute(1, 0, 2, 3)
                              .contiguous())

    # 4) Return-shaped outputs.
    unsqueeze_6 = saved_mean_1d.view(1, channels, 1, 1).contiguous()
    invstd_out = invstd_1d.contiguous()

    # 5) Update running_mean / running_var via copy_ (matches
    #    torch.batch_norm training path: population variance correction).
    if elements_per_channel > 1:
        var_correction = elements_per_channel / (elements_per_channel - 1)
    else:
        var_correction = 1.0
    new_mean = running_mean * (1.0 - MOMENTUM) + saved_mean_1d * MOMENTUM
    new_var = running_var * (1.0 - MOMENTUM) + variance_1d * var_correction * MOMENTUM
    torch.ops.aten.copy_(running_mean, new_mean)
    torch.ops.aten.copy_(running_var, new_var)

    return cat, invstd_out, relu, unsqueeze_6, running_mean, running_var
