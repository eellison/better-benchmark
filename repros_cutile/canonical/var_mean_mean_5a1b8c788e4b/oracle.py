"""cuTile port of var_mean_mean_5a1b8c788e4b: MobileNetV3 training-BN + ReLU
+ spatial mean, 3-kernel plan mirroring Triton.

1. `_partial_stats_kernel`: reduces channels-last input [N*H*W, C] into
   per-channel partial (sum, sum_sq) over BLOCK_R chunks.
2. `_finalize_stats_kernel`: combines chunk partials into (mean, var, invstd),
   stores saved_mean/invstd, and updates running_mean/running_var in-kernel
   with momentum=0.1 and correction=1.0000024912370735.
3. `_relu_spatial_mean_kernel`: applies fp32 affine, bf16 rounding, ReLU,
   writes the full activation, then reduces to per-(n, c) spatial mean.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
ONE_MINUS_MOMENTUM = 0.9
UNBIAS_CORRECTION = 1.0000024912370735


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_stats_kernel(
    x_ptr,       # bf16 [E, C]  (permuted to channels-innermost, contiguous)
    psum_ptr,    # f32  [num_chunks, C]
    psq_ptr,     # f32  [num_chunks, C]
    E: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(
        x_ptr, index=(chunk, cblk), shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    e_arange = ct.arange(BLOCK_R, dtype=ct.int32)
    e_global = chunk * BLOCK_R + e_arange
    e_valid = e_global < E
    e_valid_2d = ct.reshape(e_valid, (BLOCK_R, 1))
    zero = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.float32)
    x_masked = ct.where(e_valid_2d, x_f, zero)

    sums = ct.sum(x_masked, axis=0, keepdims=True)
    sums_sq = ct.sum(x_masked * x_masked, axis=0, keepdims=True)

    ct.store(psum_ptr, index=(chunk, cblk), tile=sums)
    ct.store(psq_ptr, index=(chunk, cblk), tile=sums_sq)


@ct.kernel
def _finalize_stats_kernel(
    psum_ptr,           # f32 [num_chunks, C]
    psq_ptr,            # f32 [num_chunks, C]
    running_mean_ptr,   # f32 [C]  (read then updated)
    running_var_ptr,    # f32 [C]  (read then updated)
    saved_mean_ptr,     # f32 [C]
    invstd_ptr,         # f32 [C]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_E: ct.Constant[float],
    EPS_: ct.Constant[float],
    UNBIAS: ct.Constant[float],
):
    cblk = ct.bid(0)

    psum = ct.load(
        psum_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    psq = ct.load(
        psq_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = chunks < NUM_CHUNKS
    chunk_mask_2d = ct.reshape(chunk_mask, (BLOCK_CHUNKS, 1))
    zero = ct.full((BLOCK_CHUNKS, BLOCK_C), 0.0, dtype=ct.float32)

    total = ct.sum(ct.where(chunk_mask_2d, psum, zero), axis=0)
    total_sq = ct.sum(ct.where(chunk_mask_2d, psq, zero), axis=0)

    mean = total * INV_E
    ex2 = total_sq * INV_E
    diff = ex2 - mean * mean
    zero_c = ct.full((BLOCK_C,), 0.0, dtype=ct.float32)
    var = ct.where(diff < zero_c, zero_c, diff)
    invstd = ct.rsqrt(var + EPS_)

    old_mean = ct.load(running_mean_ptr, index=(cblk,), shape=(BLOCK_C,))
    old_var = ct.load(running_var_ptr, index=(cblk,), shape=(BLOCK_C,))
    new_mean = old_mean * 0.9 + mean * 0.1
    corrected_var = var * UNBIAS
    new_var = old_var * 0.9 + corrected_var * 0.1

    ct.store(saved_mean_ptr, index=(cblk,), tile=mean)
    ct.store(invstd_ptr, index=(cblk,), tile=invstd)
    ct.store(running_mean_ptr, index=(cblk,), tile=new_mean)
    ct.store(running_var_ptr, index=(cblk,), tile=new_var)


@ct.kernel
def _relu_spatial_mean_kernel(
    x_ptr,               # bf16 [N*HW_PAD, C]  (zero-padded, N contiguous rows of HW_PAD each)
    weight_ptr,          # f32 [C]
    bias_ptr,            # f32 [C]
    saved_mean_ptr,      # f32 [C]
    invstd_ptr,          # f32 [C]
    activation_ptr,      # bf16 [N*HW_PAD, C]  (padded output)
    spatial_mean_ptr,    # bf16 [N, C]
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_HW: ct.Constant[float],
):
    n = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(x_ptr, index=(n, cblk), shape=(HW_PAD, BLOCK_C))
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(saved_mean_ptr, index=(cblk,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(cblk,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(cblk,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(cblk,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    scaled = normalized * weight_2d
    affine = scaled + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.full((HW_PAD, BLOCK_C), 0.0, dtype=ct.bfloat16)
    relu = ct.where(affine_bf16 < zero_bf16, zero_bf16, affine_bf16)

    ct.store(activation_ptr, index=(n, cblk), tile=relu)

    hw_arange = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_arange < HW
    hw_valid_2d = ct.reshape(hw_valid, (HW_PAD, 1))
    zero_f = ct.full((HW_PAD, BLOCK_C), 0.0, dtype=ct.float32)
    relu_f = ct.astype(relu, ct.float32)
    relu_masked = ct.where(hw_valid_2d, relu_f, zero_f)
    pooled = ct.sum(relu_masked, axis=0, keepdims=True) * INV_HW
    ct.store(spatial_mean_ptr, index=(n, cblk), tile=ct.astype(pooled, ct.bfloat16))


@oracle_impl(hardware="B200", point="765e4345",
             BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=1024, ROW_BLOCK=8)
@oracle_impl(hardware="B200", point="0cb3f08a",
             BLOCK_R=2048, BLOCK_C=8, BLOCK_HW=1024, ROW_BLOCK=8)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, BLOCK_HW: int, ROW_BLOCK: int):
    x, running_mean, running_var, weight, bias = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    device = x.device

    assert c % BLOCK_C == 0, f"c={c} not divisible by BLOCK_C={BLOCK_C}"

    num_chunks = (e + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_power_of_2(num_chunks)
    hw_pad = _next_power_of_2(hw)

    # x is channels-last [N,C,H,W] with stride[C]==1; permute(0,2,3,1) is contiguous.
    x_flat = x.permute(0, 2, 3, 1).reshape(e, c)  # bf16 [E, C] contiguous

    saved_mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    activation = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()), device=device, dtype=torch.bfloat16,
    )
    spatial_mean = torch.empty_strided(
        (n, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_chunks, c), device=device, dtype=torch.float32)
    partial_sq = torch.empty((num_chunks, c), device=device, dtype=torch.float32)

    # Padded [N, HW_PAD, C] input and output buffers for the epilogue.
    padded_x = torch.zeros((n, hw_pad, c), device=device, dtype=torch.bfloat16)
    padded_x[:, :hw, :].copy_(x_flat.view(n, hw, c))
    padded_x_2d = padded_x.view(n * hw_pad, c)

    padded_activation = torch.empty((n, hw_pad, c), device=device, dtype=torch.bfloat16)
    padded_activation_2d = padded_activation.view(n * hw_pad, c)

    saved_mean_flat = saved_mean.view(c)
    invstd_flat = invstd.view(c)
    spatial_mean_2d = spatial_mean.view(n, c)

    stream = torch.cuda.current_stream()

    # 1) Partial per-channel stats over [E, C].
    ct.launch(
        stream, (num_chunks, c // BLOCK_C, 1), _partial_stats_kernel,
        (x_flat, partial_sum, partial_sq, e, BLOCK_R, BLOCK_C),
    )

    # 2) Finalize mean/var/invstd + running-stat update in-kernel.
    ct.launch(
        stream, (c // BLOCK_C, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sq, running_mean, running_var,
         saved_mean_flat, invstd_flat,
         num_chunks, block_chunks, BLOCK_C,
         1.0 / float(e), EPS, UNBIAS_CORRECTION),
    )

    # 3) Affine + bf16 round + ReLU + spatial mean epilogue.
    ct.launch(
        stream, (n, c // BLOCK_C, 1), _relu_spatial_mean_kernel,
        (padded_x_2d, weight, bias, saved_mean_flat, invstd_flat,
         padded_activation_2d, spatial_mean_2d,
         hw, hw_pad, BLOCK_C, 1.0 / float(hw)),
    )

    # Copy the first HW rows of the padded activation into the channels-last
    # activation output (permute(0,2,3,1) of a channels-last tensor is
    # contiguous, so this is a straight memcpy of the valid region).
    activation.permute(0, 2, 3, 1).view(n, hw, c).copy_(padded_activation[:, :hw, :])

    return saved_mean, invstd, activation, spatial_mean, running_mean, running_var
