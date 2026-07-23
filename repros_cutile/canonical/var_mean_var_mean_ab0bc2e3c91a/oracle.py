"""cuTile port of var_mean_var_mean_ab0bc2e3c91a: visformer two-stage BN.

Two BN steps over [N, C, H, W] reducing to [C]. We use a cuTile kernel for
the two channel reductions (per-channel mean + variance) in one pass; the
affine and residual epilogue is torch. `copy_` updates running stats.
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
    def _reduce_kernel(
        x_ptr,      # f32 [C, NHW_padded]
        mean_ptr,   # f32 [C]
        var_ptr,    # f32 [C]
    ):
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


def _channel_mean_var(x_f32, C, NHW, BLOCK, kernel, device):
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
    return mean, var


def _run(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     arg6_1, arg7_1, arg8_1, arg9_1) = inputs
    device = arg0_1.device
    N, C, H, W = (int(d) for d in arg0_1.shape)
    NHW = N * H * W
    BLOCK = _next_pow2(NHW)
    kernel = _make_kernel(NHW, BLOCK)

    x0_f32 = arg0_1.float()
    mean0, var0 = _channel_mean_var(x0_f32, C, NHW, BLOCK, kernel, device)

    mean0_b = mean0.view(1, C, 1, 1)
    var0_b = var0.view(1, C, 1, 1)
    invstd0 = torch.rsqrt(var0_b + EPS)
    normalized = (x0_f32 - mean0_b) * invstd0
    scaled0 = normalized * arg3_1.view(1, C, 1, 1) + arg4_1.view(1, C, 1, 1)
    bf16_first = scaled0.to(torch.bfloat16)

    add_1 = mean0 * 0.1 + arg1_1 * 0.9
    add_2 = var0 * 1.0001594642002871 * 0.1 + arg2_1 * 0.9

    add_4 = bf16_first.float() + arg5_1
    mean1, var1 = _channel_mean_var(add_4, C, NHW, BLOCK, kernel, device)
    mean1_b = mean1.view(1, C, 1, 1)
    var1_b = var1.view(1, C, 1, 1)
    invstd1 = torch.rsqrt(var1_b + EPS)
    normalized1 = (add_4 - mean1_b) * invstd1
    scaled1 = normalized1 * arg8_1.view(1, C, 1, 1) + arg9_1.view(1, C, 1, 1)
    bf16_second = scaled1.to(torch.bfloat16)

    add_6 = mean1 * 0.1 + arg6_1 * 0.9
    add_7 = var1 * 1.0001594642002871 * 0.1 + arg7_1 * 0.9

    arg1_1.copy_(add_1)
    arg2_1.copy_(add_2)
    arg6_1.copy_(add_6)
    arg7_1.copy_(add_7)

    unsqueeze_10 = mean1.view(1, C, 1, 1)
    return (mean0_b, invstd0, add_4, invstd1.view(C), bf16_second,
            unsqueeze_10, arg1_1, arg2_1, arg6_1, arg7_1)


@oracle_impl(hardware="B200", point="c9bb6ab0")
@oracle_impl(hardware="B200", point="7b85be27")
@oracle_impl(hardware="B200", point="ce0e5e3c")
def oracle_forward(inputs):
    return _run(inputs)
