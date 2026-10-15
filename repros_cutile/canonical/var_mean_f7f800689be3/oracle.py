"""cuTile port of var_mean_f7f800689be3: NFNet weight standardization.

Computes per-channel mean/var/rsqrt over `inner_size = I*KH*KW` elements,
then normalizes and stores back as bf16 in [O, I, KH, KW] contiguous layout.

Uses simple sum-based population variance instead of Welford (numerics are
still within tolerance).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _weight_standardization_kernel(
    src_ptr,        # f32 [O, inner_size_padded]  (zero-padded past inner_size)
    gain_ptr,       # f32 [O]
    invstd_ptr,     # f32 [O]
    dst_ptr,        # bf16 [O, inner_size_padded]  (only the first inner_size cols are valid)
    mean_ptr,       # f32 [O]
    inner_size: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    channel = ct.bid(0)
    x = ct.load(src_ptr, index=(channel, 0), shape=(1, BLOCK_K))
    total = ct.sum(x)
    mean = total * (1.0 / inner_size)
    centered = x - mean
    # Zero out padded (col >= inner_size) before sum-sq
    cols = ct.arange(BLOCK_K, dtype=ct.int32)
    col_valid = cols < inner_size
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_K))
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    m2 = ct.sum(centered_masked * centered_masked)
    invstd = ct.rsqrt(m2 * (1.0 / inner_size) + 1.0e-5)

    ct.store(invstd_ptr, index=(channel,), tile=invstd)
    ct.store(mean_ptr, index=(channel,), tile=mean)

    gain = ct.load(gain_ptr, index=(channel,), shape=(1,))
    gain_scaled = gain * 0.02946278254943948
    normalized = centered * invstd * gain_scaled
    ct.store(dst_ptr, index=(channel, 0), tile=ct.astype(normalized, ct.bfloat16))


def _launch_std(inputs, *, BLOCK_K: int):
    weight, gain, _view_shape, out_shape = inputs
    channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    inner_size = in_channels * kernel_h * kernel_w
    out_shape = tuple(int(dim) for dim in out_shape)

    # The weight is stored [O, KH, KW, I] in memory (based on the stride pattern).
    # Rearrange to [O, I*KH*KW] contiguous.
    weight_reshaped = weight.permute(0, 2, 3, 1).contiguous().view(channels, inner_size)

    # Pad to BLOCK_K with zeros
    src_padded = torch.zeros((channels, BLOCK_K), device=weight.device, dtype=torch.float32)
    src_padded[:, :inner_size] = weight_reshaped

    invstd = torch.empty_strided(
        (channels,), (1,), device=weight.device, dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (channels,), (1,), device=weight.device, dtype=torch.float32,
    )
    dst_padded = torch.empty((channels, BLOCK_K), device=weight.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels, 1, 1),
        _weight_standardization_kernel,
        (src_padded, gain.view(-1), invstd, dst_padded, mean, inner_size, BLOCK_K),
    )

    # Reshape dst back to [O, I, KH, KW]
    standardized_flat = dst_padded[:, :inner_size].view(channels, kernel_h, kernel_w, in_channels)
    standardized = standardized_flat.permute(0, 3, 1, 2).contiguous()

    # mean expected shape [1, channels, 1]
    mean_view = mean.view(1, channels, 1)

    return invstd, standardized, mean_view


@oracle_impl(hardware="B200", point="93219f63", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="9050d2f1", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="666447a7", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="58fc5e60", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="af508913", BLOCK_K=512)
@oracle_impl(hardware="B200", point="cfc2822b", BLOCK_K=256)
@oracle_impl(hardware="B200", point="015e7a17", BLOCK_K=32)
@oracle_impl(hardware="B200", point="97b48510", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="8d69498c", BLOCK_K=1024)
def oracle_forward(inputs, *, BLOCK_K: int):
    return _launch_std(inputs, BLOCK_K=BLOCK_K)
