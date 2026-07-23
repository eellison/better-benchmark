"""cuTile port of sum_sum_sum_fb3a1658dadb: masked dual BN-backward.

Torch computes the source tensor (add + mask + where) and channel-wise
reductions. cuTile computes the dense bf16 epilogue for each BN branch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 6.228077168367346e-07


@ct.kernel
def _bn_epilogue_channel_kernel(
    source_ptr,     # f32 [C, NHW]
    centered_ptr,   # f32 [C, NHW]
    mean_source_ptr,# f32 [C]
    coeff_ptr,      # f32 [C]
    scale_ptr,      # f32 [C]
    out_ptr,        # bf16 [C, NHW]
    NHW: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    source = ct.load(source_ptr, index=(c, 0), shape=(1, BLOCK_R),
                     padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(centered_ptr, index=(c, 0), shape=(1, BLOCK_R),
                       padding_mode=ct.PaddingMode.ZERO)
    mean_s = ct.load(mean_source_ptr, index=(c,), shape=(1,))
    coeff = ct.load(coeff_ptr, index=(c,), shape=(1,))
    scale = ct.load(scale_ptr, index=(c,), shape=(1,))
    mean_2d = ct.reshape(mean_s, (1, 1))
    coeff_2d = ct.reshape(coeff, (1, 1))
    scale_2d = ct.reshape(scale, (1, 1))

    tmp = source - centered * coeff_2d - mean_2d
    out_f = tmp * scale_2d
    out = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(c, 0), tile=out)


def _run(inputs):
    x0, x1, mask, full, act0, mean0, gamma0, beta0, act1, mean1, gamma1, beta1 = inputs
    device = x0.device
    n, c, h, w = x0.shape

    # Compute `source` tensor = where(mask <= 0, full, bf16(x0 + x1))
    added_bf = (x0.to(torch.float32) + x1.to(torch.float32)).to(torch.bfloat16)
    mask_f = mask.to(torch.float32)
    source_bf = torch.where(mask_f <= 0.0, full, added_bf)
    source = source_bf.to(torch.float32)

    # Channel-wise reductions.
    centered0 = act0.to(torch.float32) - mean0
    centered1 = act1.to(torch.float32) - mean1

    sum_source = source.sum(dim=[0, 2, 3])  # [C]
    dot0 = (source * centered0).sum(dim=[0, 2, 3])
    dot1 = (source * centered1).sum(dim=[0, 2, 3])

    # Finalize per-channel stats.
    inv_nhw = REDUCE_SCALE
    mean_source = sum_source * inv_nhw
    coeff0 = (dot0 * inv_nhw) * (gamma0 * gamma0)
    scale0 = gamma0 * beta0
    coeff1 = (dot1 * inv_nhw) * (gamma1 * gamma1)
    scale1 = gamma1 * beta1

    vec0 = dot0 * gamma0
    vec1 = dot1 * gamma1

    # Dense epilogue: for each element at channel c,
    # out = ((source - centered * coeff[c]) - mean_source[c]) * scale[c]
    # Since we may have non-contiguous input strides (channels-last), we work in
    # the natural NCHW indexing.
    # For the epilogue, use a per-channel cuTile kernel that processes all
    # elements belonging to one channel in a single tile.
    # source, centered0, centered1 are (N, C, H, W). Reorganize to (C, N*H*W)
    # by contiguous reshape then flatten.
    source_cn = source.permute(1, 0, 2, 3).contiguous().view(c, n * h * w)
    centered0_cn = centered0.permute(1, 0, 2, 3).contiguous().view(c, n * h * w)
    centered1_cn = centered1.permute(1, 0, 2, 3).contiguous().view(c, n * h * w)

    out0_cn = torch.empty((c, n * h * w), device=device, dtype=torch.bfloat16)
    out1_cn = torch.empty((c, n * h * w), device=device, dtype=torch.bfloat16)

    nhw = n * h * w
    block_r = 1 << (nhw - 1).bit_length()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _bn_epilogue_channel_kernel,
        (source_cn, centered0_cn, mean_source, coeff0, scale0,
         out0_cn, nhw, block_r),
    )
    ct.launch(
        stream, (c, 1, 1), _bn_epilogue_channel_kernel,
        (source_cn, centered1_cn, mean_source, coeff1, scale1,
         out1_cn, nhw, block_r),
    )

    # Materialize back into act0's / act1's stride pattern.
    out0_nchw = out0_cn.view(c, n, h, w).permute(1, 0, 2, 3).contiguous()
    out1_nchw = out1_cn.view(c, n, h, w).permute(1, 0, 2, 3).contiguous()
    out0 = torch.empty_strided(tuple(act0.shape), tuple(act0.stride()),
                               device=device, dtype=torch.bfloat16)
    out1 = torch.empty_strided(tuple(act1.shape), tuple(act1.stride()),
                               device=device, dtype=torch.bfloat16)
    out0.copy_(out0_nchw)
    out1.copy_(out1_nchw)

    return sum_source, vec0, out0, sum_source, vec1, out1


# The BLOCK must evenly divide N*C*H*W for cuTile stores. All spatial sizes are
# multiples of 1024, so we use BLOCK=1024 as a common divisor.
_POINTS = [
    "7a67de76", "32659d96", "a05f5297", "ee822c25", "947f7696",
    "4ff6585f", "123127af", "d2c78c03", "b09bc4e6", "db75ae04",
    "45fc19b9", "1f5f432a", "f3056617", "5ca11463", "5ebdcf4f",
    "0b10d4fe", "5edb39f6", "6880ff5a", "0454b2c3", "80c40bdb",
]


def _decorate(fn):
    for pt in _POINTS:
        fn = oracle_impl(hardware="B200", point=pt)(fn)
    return fn


@_decorate
def oracle_forward(inputs):
    return _run(inputs)
