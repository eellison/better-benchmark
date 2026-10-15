"""cuTile port of var_mean_ff900ef0c453: MobileNetV2 BN training + residual add.

Three cuTile kernels (matching Triton's general 3-kernel path):
  1. `_bn_partial_stats_kernel`: per-channel partial sum + sum-of-squares over
     BLOCK_R chunks of the flattened N*H*W dimension.
  2. `_bn_finalize_stats_kernel`: combines chunk partials, computes
     (mean, var, invstd), writes running_mean/running_var in-place, and stores
     saved mean/invstd.
  3. `_affine_residual_kernel`: bf16 affine + residual add + bf16 cast.

Input is NCHW contiguous (C in {24, 32, 64, 96, 160} — not always power of 2).
We use one program per channel so channel dim doesn't need power-of-2 tiling.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


RUNNING_VAR_CORRECTION = 1.0002126302360195
EPS = 1.0e-5
MOMENTUM = 0.1


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


@ct.kernel
def _bn_partial_stats_kernel(
    x_ptr,             # bf16 [C, E]
    partial_sum_ptr,   # f32  [num_chunks, C]
    partial_sum2_ptr,  # f32  [num_chunks, C]
    E: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    chunk = ct.bid(1)

    vals = ct.load(
        x_ptr, index=(channel, chunk), shape=(1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals_f = ct.astype(vals, ct.float32)

    offsets = ct.arange(BLOCK_R, dtype=ct.int32)
    valid_1d = offsets < (E - chunk * BLOCK_R)
    valid = ct.reshape(valid_1d, (1, BLOCK_R))
    zero_f = ct.full((1, BLOCK_R), 0.0, dtype=ct.float32)
    masked = ct.where(valid, vals_f, zero_f)
    s = ct.sum(masked)
    s2 = ct.sum(masked * masked)

    s_2 = ct.reshape(s, (1, 1))
    s2_2 = ct.reshape(s2, (1, 1))
    ct.store(partial_sum_ptr, index=(chunk, channel), tile=s_2)
    ct.store(partial_sum2_ptr, index=(chunk, channel), tile=s2_2)


@ct.kernel
def _bn_finalize_stats_kernel(
    partial_sum_ptr,   # f32 [num_chunks, C]
    partial_sum2_ptr,  # f32 [num_chunks, C]
    running_mean_ptr,  # f32 [C] (read-modify-write)
    running_var_ptr,   # f32 [C] (read-modify-write)
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    E: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    EPS_: ct.Constant[float],
    MOM: ct.Constant[float],
    CORR: ct.Constant[float],
):
    channel = ct.bid(0)

    sums = ct.load(
        partial_sum_ptr, index=(0, channel), shape=(BLOCK_CHUNKS, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sums2 = ct.load(
        partial_sum2_ptr, index=(0, channel), shape=(BLOCK_CHUNKS, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    offsets = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    valid_1d = offsets < NUM_CHUNKS
    valid = ct.reshape(valid_1d, (BLOCK_CHUNKS, 1))
    zero_f = ct.full((BLOCK_CHUNKS, 1), 0.0, dtype=ct.float32)
    sums = ct.where(valid, sums, zero_f)
    sums2 = ct.where(valid, sums2, zero_f)

    sum_x = ct.sum(sums)
    sum_x2 = ct.sum(sums2)
    inv_e = 1.0 / E
    mean = sum_x * inv_e
    ex2 = sum_x2 * inv_e
    var = ex2 - mean * mean

    # Clamp negative var (numerical noise) to 0 via [1]-tile compare.
    mean_1 = ct.reshape(mean, (1,))
    var_1 = ct.reshape(var, (1,))
    zero_1 = ct.full((1,), 0.0, dtype=ct.float32)
    var_1 = ct.where(var_1 < zero_1, zero_1, var_1)
    invstd_1 = ct.rsqrt(var_1 + EPS_)

    old_mean = ct.load(running_mean_ptr, index=(channel,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(channel,), shape=(1,))
    new_mean = old_mean * (1.0 - MOM) + mean_1 * MOM
    corrected_var = var_1 * CORR
    new_var = old_var * (1.0 - MOM) + corrected_var * MOM
    ct.store(running_mean_ptr, index=(channel,), tile=new_mean)
    ct.store(running_var_ptr, index=(channel,), tile=new_var)
    ct.store(mean_ptr, index=(channel,), tile=mean_1)
    ct.store(invstd_ptr, index=(channel,), tile=invstd_1)


@ct.kernel
def _affine_residual_kernel(
    x_ptr,        # bf16 [N_FLAT]
    residual_ptr, # bf16 [N_FLAT]
    mean_ptr,     # f32 [C]
    invstd_ptr,   # f32 [C]
    weight_ptr,   # f32 [C]
    bias_ptr,     # f32 [C]
    out_ptr,      # bf16 [N_FLAT]
    C_C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))

    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel_full = idxs // HW
    channel = channel_full - (channel_full // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)

    x_f = ct.astype(x, ct.float32)
    centered = x_f - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine = scaled + bias
    rounded = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    total = rounded_f + resid_f
    ct.store(out_ptr, index=(pid,), tile=ct.astype(total, ct.bfloat16))


@oracle_impl(hardware="B200", point="46331323")
@oracle_impl(hardware="B200", point="0d283d85")
@oracle_impl(hardware="B200", point="d820c4c3")
@oracle_impl(hardware="B200", point="a851217b")
@oracle_impl(hardware="B200", point="eb17bc84")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, residual = inputs
    n, c, h, w = (int(d) for d in x.shape)
    hw = h * w
    e = n * hw
    total = n * c * hw

    device = x.device

    BLOCK = 512
    BLOCK_R = 4096
    num_chunks = (e + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_pow2(num_chunks)

    stream = torch.cuda.current_stream()

    # Permute x from [N, C, H, W] to [C, N*H*W] contiguous.
    x_perm = x.permute(1, 0, 2, 3).contiguous().view(c, e)

    partial_sum = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, c), device=device, dtype=torch.float32)

    mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    mean_flat = mean.view(c)

    # 1. partial stats: grid = (C, num_chunks)
    ct.launch(
        stream,
        (c, num_chunks, 1),
        _bn_partial_stats_kernel,
        (x_perm, partial_sum, partial_sum2, e, BLOCK_R),
    )

    # 2. finalize + running-stat updates: grid = (C,)
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_finalize_stats_kernel,
        (partial_sum, partial_sum2, running_mean, running_var,
         mean_flat, invstd,
         e, num_chunks, block_chunks, EPS, MOMENTUM, RUNNING_VAR_CORRECTION),
    )

    # 3. affine + residual + bf16 cast
    x_flat = x.contiguous().view(-1)
    residual_flat = residual.contiguous().view(-1)
    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _affine_residual_kernel,
        (x_flat, residual_flat, mean_flat, invstd, weight, bias, out_flat,
         c, hw, BLOCK),
    )

    y = out_flat.view(n, c, h, w)
    return invstd, y, mean, running_mean, running_var
