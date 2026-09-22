"""cuTile port of var_mean_481c5828fa83: DenseNet BN training with 5-input cat.

Concats 5 bf16 tensors along channel, then BN over [N, H, W] axes,
followed by affine + ReLU + running-stat updates.

cuTile does per-channel mean+variance in a reduction kernel; the concat,
affine, ReLU, and copy_ updates are torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


def _make_kernel(NHW: int, BLOCK: int):
    @ct.kernel
    def _reduce_kernel(x_ptr, mean_ptr, var_ptr):
        c = ct.bid(0)
        x = ct.load(x_ptr, index=(c, 0), shape=(1, BLOCK),
                    padding_mode=ct.PaddingMode.ZERO)
        col_idx = ct.arange(BLOCK, dtype=ct.int32)
        col_mask = ct.reshape(col_idx < NHW, (1, BLOCK))
        zero = ct.full((1, BLOCK), 0.0, dtype=ct.float32)
        x_m = ct.where(col_mask, x, zero)
        total = ct.sum(x_m)
        mean = total * (1.0 / NHW)
        ct.store(mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
        centered = x - mean
        centered_m = ct.where(col_mask, centered, zero)
        var = ct.sum(centered_m * centered_m) * (1.0 / NHW)
        ct.store(var_ptr, index=(c,), tile=ct.reshape(var, (1,)))

    return _reduce_kernel


@oracle_impl(hardware="B200", point="85964560")
@oracle_impl(hardware="B200", point="4d94142f")
@oracle_impl(hardware="B200", point="e0b577eb")
@oracle_impl(hardware="B200", point="920aacff")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = inputs
    device = arg0_1.device
    cat = torch.cat([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1], dim=1)
    N, C, H, W = cat.shape
    NHW = int(N * H * W)
    BLOCK = _next_pow2(NHW)
    kernel = _make_kernel(NHW, BLOCK)

    x_f32 = cat.float()
    x_perm = x_f32.permute(1, 0, 2, 3).contiguous().view(C, NHW)
    if BLOCK != NHW:
        p = torch.zeros((C, BLOCK), device=device, dtype=torch.float32)
        p[:, :NHW].copy_(x_perm)
    else:
        p = x_perm
    mean = torch.empty((C,), device=device, dtype=torch.float32)
    var = torch.empty((C,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (C, 1, 1), kernel, (p, mean, var))

    mean_b = mean.view(1, C, 1, 1)
    var_b = var.view(1, C, 1, 1)
    invstd = torch.rsqrt(var_b + EPS)
    normalized = (x_f32 - mean_b) * invstd
    scaled = normalized * arg7_1.view(1, C, 1, 1) + arg8_1.view(1, C, 1, 1)
    relu = torch.relu(scaled.to(torch.bfloat16))

    unbiased_factor = NHW / (NHW - 1)
    add_1 = mean * 0.1 + arg5_1 * 0.9
    add_2 = var * unbiased_factor * 0.1 + arg6_1 * 0.9
    arg5_1.copy_(add_1)
    arg6_1.copy_(add_2)

    unsqueeze_6 = mean.view(1, C, 1, 1)
    return cat, invstd.view(C), relu, unsqueeze_6, arg5_1, arg6_1
