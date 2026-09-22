"""cuTile port of var_mean_e184692a514a: DenseNet concat + BN train + affine+relu.

Uses torch to cat 4 inputs then a cuTile kernel to reduce mean/var over batch
and spatial dims per channel, then a cuTile kernel to apply affine + bf16 +
relu. Non-power-of-2 spatial N is padded to a power-of-2 BLOCK with masking.
Running mean/var updated via torch aten copy_ from the oracle.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
INV_MOMENTUM = 0.9


def _make_kernels(BLOCK_N):
    @ct.kernel
    def _bn_reduce(
        x_ptr,           # f32 [C, BLOCK_N] (zero-padded past N)
        mean_ptr,        # f32 [C]
        invstd_ptr,      # f32 [C]
        var_ptr,         # f32 [C]
        N: ct.Constant[int],
    ):
        c = ct.bid(0)
        x = ct.load(x_ptr, index=(c, 0), shape=(1, BLOCK_N))
        idx = ct.arange(BLOCK_N, dtype=ct.int32)
        idx_mask = ct.reshape(idx < N, (1, BLOCK_N))
        x_masked = ct.where(idx_mask, x, 0.0)
        total = ct.sum(x_masked)
        mean = total * (1.0 / N)
        centered = x_masked - mean
        centered_masked = ct.where(idx_mask, centered, 0.0)
        sumsq = ct.sum(centered_masked * centered_masked)
        var = sumsq * (1.0 / N)
        invstd = ct.rsqrt(var + EPS)
        ct.store(mean_ptr, index=(c,), tile=ct.reshape(
            ct.full((1,), mean, dtype=ct.float32), (1,)))
        ct.store(invstd_ptr, index=(c,), tile=ct.reshape(
            ct.full((1,), invstd, dtype=ct.float32), (1,)))
        ct.store(var_ptr, index=(c,), tile=ct.reshape(
            ct.full((1,), var, dtype=ct.float32), (1,)))

    @ct.kernel
    def _bn_affine(
        x_ptr,           # f32 [C, BLOCK_N]
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,         # bf16 [C, BLOCK_N]
        normalized_ptr,  # f32 [C, BLOCK_N]
    ):
        c = ct.bid(0)
        x = ct.load(x_ptr, index=(c, 0), shape=(1, BLOCK_N))
        mean_t = ct.load(mean_ptr, index=(c,), shape=(1,))
        invstd_t = ct.load(invstd_ptr, index=(c,), shape=(1,))
        weight_t = ct.load(weight_ptr, index=(c,), shape=(1,))
        bias_t = ct.load(bias_ptr, index=(c,), shape=(1,))
        normalized = (x - mean_t) * invstd_t
        affine = normalized * weight_t + bias_t
        affine_bf = ct.astype(affine, ct.bfloat16)
        zero_bf = ct.astype(
            ct.full(shape=(1, BLOCK_N), fill_value=0.0, dtype=ct.float32),
            ct.bfloat16,
        )
        relu = ct.where(affine_bf > zero_bf, affine_bf, zero_bf)
        ct.store(out_ptr, index=(c, 0), tile=relu)
        ct.store(normalized_ptr, index=(c, 0), tile=normalized)

    return _bn_reduce, _bn_affine


def _next_pow2(n):
    x = 1
    while x < n:
        x <<= 1
    return x


_KERNEL_CACHE = {}


def _get_kernels(BLOCK_N):
    if BLOCK_N not in _KERNEL_CACHE:
        _KERNEL_CACHE[BLOCK_N] = _make_kernels(BLOCK_N)
    return _KERNEL_CACHE[BLOCK_N]


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device

    cat = torch.cat([arg0_1, arg1_1, arg2_1, arg3_1], dim=1)
    B, C, H, W = cat.shape
    N = B * H * W
    BLOCK_N = _next_pow2(N)

    x_f32 = cat.float()
    x_cn = x_f32.permute(1, 0, 2, 3).contiguous().view(C, N)
    if BLOCK_N == N:
        x_padded = x_cn
    else:
        x_padded = torch.zeros((C, BLOCK_N), device=device, dtype=torch.float32)
        x_padded[:, :N].copy_(x_cn)

    mean_1d = torch.empty((C,), device=device, dtype=torch.float32)
    invstd_1d = torch.empty((C,), device=device, dtype=torch.float32)
    var_1d = torch.empty((C,), device=device, dtype=torch.float32)

    reduce_k, affine_k = _get_kernels(BLOCK_N)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), reduce_k,
        (x_padded, mean_1d, invstd_1d, var_1d, N),
    )

    out_padded = torch.empty((C, BLOCK_N), device=device, dtype=torch.bfloat16)
    normalized_padded = torch.empty((C, BLOCK_N), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, 1, 1), affine_k,
        (x_padded, mean_1d, invstd_1d, arg6_1, arg7_1,
         out_padded, normalized_padded),
    )

    out_cn = out_padded[:, :N]
    normalized_cn = normalized_padded[:, :N]

    out = out_cn.contiguous().view(C, B, H, W).permute(1, 0, 2, 3).contiguous()
    del normalized_cn  # unused in returned outputs, but kept for correctness reasoning

    unbiased_scale = N / (N - 1)
    unbiased_var = var_1d * unbiased_scale

    new_running_mean = mean_1d * MOMENTUM + arg4_1 * INV_MOMENTUM
    new_running_var = unbiased_var * MOMENTUM + arg5_1 * INV_MOMENTUM
    copy0 = torch.ops.aten.copy_.default(arg4_1, new_running_mean)
    copy1 = torch.ops.aten.copy_.default(arg5_1, new_running_var)

    squeeze_1 = invstd_1d
    unsqueeze_6 = mean_1d.view(1, C, 1, 1)

    return cat, squeeze_1, out, unsqueeze_6, copy0, copy1


@oracle_impl(hardware="B200", point="44053229")
@oracle_impl(hardware="B200", point="3c0535dc")
@oracle_impl(hardware="B200", point="7bdf6797")
@oracle_impl(hardware="B200", point="31bfd422")
def oracle_forward(inputs):
    return _run(inputs)
