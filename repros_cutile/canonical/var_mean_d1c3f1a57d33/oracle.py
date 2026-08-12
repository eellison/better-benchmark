"""cuTile port of var_mean_d1c3f1a57d33 (SCHEDULER_FUSION): NFNet weight
standardization.

For each output channel c, compute mean/var over the inner (in_c*kh*kw) dim,
apply eps=1e-5 rsqrt, scale by `gain[c] * 0.02946278254943948`, and store
the normalized weights in bf16 with contiguous target layout.

Weight is channels-last (stride [inner_size, 1, kw*in_c, in_c]) — we
`.contiguous()` it to (channels, in_c, kh, kw) contiguous before use so that
the kernel can walk a plain (channels, inner_size) tile.

Inner sizes here are non-power-of-2 (27, 144, 288, 576, 1152). Round up to
next power of 2 for the tile; mask cols>=inner_size to 0 for reductions and
skip stores past inner_size by padding the output to next_po2 and copying.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _weight_standardization_kernel(
    weight_ptr,   # bf16 [channels, inner_size]
    gain_ptr,     # bf16 [channels]
    out_ptr,      # bf16 [channels, PAD_K]  (padded)
    inner_size: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    channel = ct.bid(0)
    x_bf16 = ct.load(weight_ptr, index=(channel, 0), shape=(1, BLOCK_K),
                     padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf16, ct.float32)
    col_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    col_mask = ct.reshape(col_idx, (1, BLOCK_K)) < inner_size

    x_masked = ct.where(col_mask, x, 0.0)
    sum_x = ct.sum(x_masked, axis=1, keepdims=True)
    inv_n = 1.0 / inner_size
    mean = sum_x * inv_n
    # var = E[x^2] - mean^2 for population variance (correction=0)
    sum_x2 = ct.sum(ct.where(col_mask, x * x, 0.0), axis=1, keepdims=True)
    m2 = sum_x2 - sum_x * mean
    invstd = ct.rsqrt(m2 * inv_n + 1.0e-5)

    gain_val = ct.load(gain_ptr, index=(channel,), shape=(1,))
    gain_f = ct.astype(gain_val, ct.float32)
    gain_scaled = gain_f * 0.02946278254943948
    gain_scaled_2d = ct.reshape(gain_scaled, (1, 1))

    centered = x - mean
    normalized = centered * invstd
    y = normalized * gain_scaled_2d
    ct.store(out_ptr, index=(channel, 0), tile=ct.astype(y, ct.bfloat16))


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="fec5bff7", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="328a377c", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="25be6f49", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="503f9220", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="42d3582f", BLOCK_K=512)
@oracle_impl(hardware="B200", point="fdcbb92a", BLOCK_K=256)
@oracle_impl(hardware="B200", point="894d940e", BLOCK_K=32)
@oracle_impl(hardware="B200", point="a866684b", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="13b6121f", BLOCK_K=1024)
def oracle_forward(inputs, *, BLOCK_K):
    weight, gain, _view_shape, out_shape = inputs
    channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    inner_size = in_channels * kernel_h * kernel_w
    out_shape = tuple(int(dim) for dim in out_shape)
    out_stride = (inner_size, kernel_h * kernel_w, kernel_w, 1)

    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=weight.device,
        dtype=torch.bfloat16,
    )

    # Reshape weight to (channels, inner_size) contiguous. weight is
    # channels-last strided so we .contiguous() then view.
    weight_contig = weight.contiguous()
    weight_2d = weight_contig.view(channels, inner_size)

    gain_1d = gain.view(-1)

    pad_k = _next_power_of_2(inner_size)
    if BLOCK_K < pad_k:
        # BLOCK_K too small — bump up to cover the whole row
        pad_k = _next_power_of_2(inner_size)
    if pad_k == inner_size:
        out_2d = out.view(channels, inner_size)
    else:
        out_padded = torch.empty((channels, pad_k), device=weight.device,
                                 dtype=torch.bfloat16)
        out_2d = out_padded

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, 1, 1),
        _weight_standardization_kernel,
        (weight_2d, gain_1d, out_2d, inner_size, pad_k),
    )
    if pad_k != inner_size:
        out.view(channels, inner_size).copy_(out_2d[:, :inner_size])
    return out
