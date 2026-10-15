"""cuTile port of sum_sum_fdba1485ca74: DenseNet BN-backward + slice-add.

Matches Triton's 3-kernel structure:
  1. _partial_reduce_kernel: per (C, K_block), computes partial sum(grad) and
     partial sum(grad * centered) over K = N*HW elements.
  2. _finalize_kernel: per C, sums partials -> sum_out, dot_out, mul8_out.
  3. _epilogue_kernel: per (N, C), computes elementwise BN-backward dense
     tensor and the last-32-channel residual slice-add.

We view inputs as (N, C, HW) with HW padded to power-of-2 for tile-friendly
loads; masks handle OOB elements.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 192
H = 56
W = 56
HW = H * W  # 3136
K_TOTAL = N * HW  # 12544
SCALE = 7.971938775510203e-05
SLICE_START = 160
SLICE_C = 32  # 192 - 160


@ct.kernel
def _partial_reduce_kernel(
    where_bf_ptr,        # bf16 [N, C, HW_PAD]   the resolved where(mask<=0, fill, rhs)
    centered_f_ptr,      # f32  [N, C, HW_PAD]
    partial_sum_ptr,     # f32  [C, NUM_N_BLOCKS] partial_sum[c, n]
    partial_dot_ptr,     # f32  [C, NUM_N_BLOCKS]
    HW_C: ct.Constant[int],
    HW_PAD: ct.Constant[int],
):
    # One pid = one (c, n) pair. Each program reduces over HW_PAD elements
    # (with a mask for the true HW).
    c = ct.bid(0)
    n = ct.bid(1)

    where_bf = ct.load(where_bf_ptr, index=(n, c, 0),
                       shape=(1, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    centered_f = ct.load(centered_f_ptr, index=(n, c, 0),
                         shape=(1, 1, HW_PAD),
                         padding_mode=ct.PaddingMode.ZERO)
    where_f = ct.astype(where_bf, ct.float32)

    hw_idx = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_idx < HW_C
    hw_valid_3d = ct.reshape(hw_valid, (1, 1, HW_PAD))
    zero_3d = ct.zeros((1, 1, HW_PAD), dtype=ct.float32)

    where_masked = ct.where(hw_valid_3d, where_f, zero_3d)
    dot_masked = ct.where(hw_valid_3d, where_f * centered_f, zero_3d)
    p_sum = ct.sum(where_masked)
    p_dot = ct.sum(dot_masked)

    ct.store(partial_sum_ptr, index=(c, n), tile=ct.reshape(p_sum, (1, 1)))
    ct.store(partial_dot_ptr, index=(c, n), tile=ct.reshape(p_dot, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,  # f32 [C, N]
    partial_dot_ptr,  # f32 [C, N]
    invstd_ptr,       # f32 [C]
    sum_out_ptr,      # f32 [C]
    dot_out_ptr,      # f32 [C]
    mul8_out_ptr,     # f32 [C]
    N_PAD: ct.Constant[int],
    N_ACTUAL: ct.Constant[int],
):
    c = ct.bid(0)
    # Load (1, N_PAD) row for this channel.
    sums = ct.load(partial_sum_ptr, index=(c, 0),
                   shape=(1, N_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    dots = ct.load(partial_dot_ptr, index=(c, 0),
                   shape=(1, N_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    n_idx = ct.arange(N_PAD, dtype=ct.int32)
    n_valid = ct.reshape(n_idx < N_ACTUAL, (1, N_PAD))
    zero_2d = ct.zeros((1, N_PAD), dtype=ct.float32)
    sums_m = ct.where(n_valid, sums, zero_2d)
    dots_m = ct.where(n_valid, dots, zero_2d)
    sum_value = ct.sum(sums_m)
    dot_value = ct.sum(dots_m)

    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(dot_out_ptr, index=(c,), tile=ct.reshape(dot_value, (1,)))
    ct.store(mul8_out_ptr, index=(c,),
             tile=ct.reshape(dot_value * ct.reshape(invstd, ()), (1,)))


@ct.kernel
def _bn_epilogue_kernel(
    where_bf_ptr,      # bf16 [N, C, HW_PAD]  the resolved (mask/fill/rhs) tile
    centered_f_ptr,    # f32  [N, C, HW_PAD]
    mean_ptr,          # f32  [C]
    invstd_ptr,        # f32  [C]
    weight_ptr,        # f32  [C]
    sum_ptr,           # f32  [C]
    dot_ptr,           # f32  [C]
    dense_out_ptr,     # bf16 [N, C, HW_PAD]
    HW_C: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    n = ct.bid(0)
    c = ct.bid(1)

    where_bf = ct.load(where_bf_ptr, index=(n, c, 0),
                       shape=(1, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    centered_f = ct.load(centered_f_ptr, index=(n, c, 0),
                         shape=(1, 1, HW_PAD),
                         padding_mode=ct.PaddingMode.ZERO)
    where_f = ct.astype(where_bf, ct.float32)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum_value = ct.load(sum_ptr, index=(c,), shape=(1,))
    dot_value = ct.load(dot_ptr, index=(c,), shape=(1,))

    mean_3d = ct.reshape(mean, (1, 1, 1))
    invstd_3d = ct.reshape(invstd, (1, 1, 1))
    weight_3d = ct.reshape(weight, (1, 1, 1))
    sum_3d = ct.reshape(sum_value, (1, 1, 1))
    dot_3d = ct.reshape(dot_value, (1, 1, 1))

    mean_term = sum_3d * SCALE_C
    dot_scaled = dot_3d * SCALE_C
    invstd_sq = invstd_3d * invstd_3d
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd_3d * weight_3d

    after_variance = where_f - centered_f * variance_term
    after_mean = after_variance - mean_term
    dense_f32 = after_mean * output_scale
    dense_bf = ct.astype(dense_f32, ct.bfloat16)
    ct.store(dense_out_ptr, index=(n, c, 0), tile=dense_bf)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="611efccd", BLOCK_K=8192)
def oracle_forward(inputs, *, BLOCK_K):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 = inputs
    device = arg2.device

    # Step 1: precompute resolved where and centered tensors on-device.
    #   where_result = where(arg2 <= 0, arg3, arg4)   (bf16 [N, C, H, W])
    #   centered_f   = arg5.f32 - arg6                 (f32 [N, C, H, W])
    # These are the inputs Triton implicitly recomputes in every kernel.
    where_result = torch.where(arg2 <= 0, arg3, arg4).contiguous()
    centered_f = (arg5.to(torch.float32) - arg6).contiguous()

    # Flatten HW to 1D for cuTile-friendly tile loading. HW=3136 -> pad to 4096.
    hw_pad = _next_power_of_2(HW)
    n_pad = _next_power_of_2(N)

    # Views with padded HW dim. We reshape (N, C, H, W) to (N, C, HW) and let
    # cuTile pad_mode ZERO handle the OOB elements to hw_pad.
    where_view = where_result.view(N, C, HW)
    centered_view = centered_f.view(N, C, HW)

    mean_1d = arg6.view(C)
    invstd_1d = arg7.view(C)
    weight_1d = arg8.view(C)

    # Kernel 1: partial reductions over each (c, n) pair.
    partial_sum = torch.empty((C, N), device=device, dtype=torch.float32)
    partial_dot = torch.empty((C, N), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, N, 1), _partial_reduce_kernel,
        (where_view, centered_view, partial_sum, partial_dot, HW, hw_pad),
    )

    # Kernel 2: finalize per-channel sums.
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    mul8_out = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, sum_out, dot_out, mul8_out,
         n_pad, N),
    )

    # Kernel 3: BN-backward epilogue.
    dense_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_view = dense_bf16.view(N, C, HW)
    ct.launch(
        stream, (N, C, 1), _bn_epilogue_kernel,
        (where_view, centered_view, mean_1d, invstd_1d, weight_1d,
         sum_out, dot_out, dense_view, HW, hw_pad, SCALE),
    )

    # Slice-add via torch (metadata-only slices + bf16 add).
    resid = arg0[:, SLICE_START:C] + arg1[:, SLICE_START:C]
    add_out = resid + dense_bf16[:, SLICE_START:C]

    return sum_out, mul8_out, dense_bf16, add_out
