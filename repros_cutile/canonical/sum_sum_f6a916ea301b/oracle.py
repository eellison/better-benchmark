"""cuTile port of sum_sum_f6a916ea301b: GhostNet BN backward (slice + add + reduce).

Matches Triton's 3-kernel structure:
  1. _partial_reduce_kernel: slice + bf16(add) + centered, produces partial
     per-channel sums and dot sums.
  2. _finalize_kernel: reduce partials, compute channel coefficients.
  3. _epilogue_kernel: elementwise BN-backward, write bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
FULL_C = 40
C = 20
H = 28
W = 28
HW = H * W
R = N * HW
NUMEL = R * C
INV_R = 2.4912308673469386e-06


@ct.kernel
def _partial_reduce_kernel(
    slice_base_ptr,     # bf16 (R, FULL_C)  — full 40-channel source
    add_rhs_ptr,        # bf16 (R, C)       — 20-channel residual (dense)
    centered_src_ptr,   # bf16 (R, C)
    mean_ptr,           # f32 (C,)
    partial_sum_ptr,    # f32 (num_r_tiles, C)
    partial_dot_ptr,    # f32 (num_r_tiles, C)
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r_block = ct.bid(0)
    c_block = ct.bid(1)

    # Load slice: (BLOCK_R, BLOCK_C) from the first BLOCK_C channels.
    lhs = ct.load(slice_base_ptr, index=(r_block, c_block),
                  shape=(BLOCK_R, BLOCK_C))
    rhs = ct.load(add_rhs_ptr, index=(r_block, c_block),
                  shape=(BLOCK_R, BLOCK_C))
    x = ct.load(centered_src_ptr, index=(r_block, c_block),
                shape=(BLOCK_R, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))

    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_bf = ct.astype(lhs_f + rhs_f, ct.bfloat16)
    value = ct.astype(add_bf, ct.float32)
    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = x_f - mean_2d
    dot = value * centered

    sum_v = ct.sum(value, axis=0)
    dot_v = ct.sum(dot, axis=0)
    ct.store(partial_sum_ptr, index=(r_block, c_block),
             tile=ct.reshape(sum_v, (1, BLOCK_C)))
    ct.store(partial_dot_ptr, index=(r_block, c_block),
             tile=ct.reshape(dot_v, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (num_r_tiles, C)
    partial_dot_ptr,   # f32 (num_r_tiles, C)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    sum_out_ptr,       # f32 (C,)
    vec_out_ptr,       # f32 (C,) = dot * invstd
    mean_term_ptr,     # f32 (C,) = sum * INV_R
    dot_coeff_ptr,     # f32 (C,) = dot * INV_R * invstd^2
    out_scale_ptr,     # f32 (C,) = invstd * weight
    NUM_R_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_R_C: ct.Constant[float],
):
    c_block = ct.bid(0)
    total_sum = ct.zeros((BLOCK_C,), dtype=ct.float32)
    total_dot = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for t in range(NUM_R_TILES):
        s = ct.load(partial_sum_ptr, index=(t, c_block), shape=(1, BLOCK_C))
        d = ct.load(partial_dot_ptr, index=(t, c_block), shape=(1, BLOCK_C))
        total_sum = total_sum + ct.reshape(s, (BLOCK_C,))
        total_dot = total_dot + ct.reshape(d, (BLOCK_C,))

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    dot_scaled = total_dot * INV_R_C
    invstd_sq = invstd * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=total_sum)
    ct.store(vec_out_ptr, index=(c_block,), tile=total_dot * invstd)
    ct.store(mean_term_ptr, index=(c_block,), tile=total_sum * INV_R_C)
    ct.store(dot_coeff_ptr, index=(c_block,), tile=dot_scaled * invstd_sq)
    ct.store(out_scale_ptr, index=(c_block,), tile=invstd * weight)


@ct.kernel
def _epilogue_kernel(
    slice_base_ptr,   # bf16 (R, FULL_C)
    add_rhs_ptr,      # bf16 (R, C)
    centered_src_ptr, # bf16 (R, C)
    mean_ptr,         # f32 (C,)
    mean_term_ptr,    # f32 (C,)
    dot_coeff_ptr,    # f32 (C,)
    out_scale_ptr,    # f32 (C,)
    out_ptr,          # bf16 (R, C)
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r_block = ct.bid(0)
    c_block = ct.bid(1)
    lhs = ct.load(slice_base_ptr, index=(r_block, c_block),
                  shape=(BLOCK_R, BLOCK_C))
    rhs = ct.load(add_rhs_ptr, index=(r_block, c_block),
                  shape=(BLOCK_R, BLOCK_C))
    x = ct.load(centered_src_ptr, index=(r_block, c_block),
                shape=(BLOCK_R, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_term = ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,))
    dot_coeff = ct.load(dot_coeff_ptr, index=(c_block,), shape=(BLOCK_C,))
    out_scale = ct.load(out_scale_ptr, index=(c_block,), shape=(BLOCK_C,))

    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_bf = ct.astype(lhs_f + rhs_f, ct.bfloat16)
    value = ct.astype(add_bf, ct.float32)
    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    dot_coeff_2d = ct.reshape(dot_coeff, (1, BLOCK_C))
    out_scale_2d = ct.reshape(out_scale, (1, BLOCK_C))

    centered = x_f - mean_2d
    adjusted = value - centered * dot_coeff_2d - mean_term_2d
    out = adjusted * out_scale_2d
    ct.store(out_ptr, index=(r_block, c_block),
             tile=ct.astype(out, ct.bfloat16))


C_PAD = 32  # next pow2 >= C=20


@oracle_impl(hardware="B200", point="537609ac", BLOCK_R=512, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # NHWC-flatten and pad the C axis to BLOCK_C so that tile shapes are pow-2.
    slice_flat_c40 = arg0_1.permute(0, 2, 3, 1).contiguous().view(R, FULL_C)
    resid_flat_c20 = arg1_1.permute(0, 2, 3, 1).contiguous().view(R, C)
    x_flat_c20 = arg2_1.permute(0, 2, 3, 1).contiguous().view(R, C)
    # Pad the residual/x/mean to BLOCK_C along last dim; the slice already has 40>32
    # channels but we only want the first BLOCK_C to be usable — actually we need
    # first C to be data and rest zero, so pad slice to BLOCK_C too.
    slice_pad = torch.zeros((R, BLOCK_C), device=device, dtype=torch.bfloat16)
    slice_pad[:, :C] = slice_flat_c40[:, :C]
    resid_pad = torch.zeros((R, BLOCK_C), device=device, dtype=torch.bfloat16)
    resid_pad[:, :C] = resid_flat_c20
    x_pad = torch.zeros((R, BLOCK_C), device=device, dtype=torch.bfloat16)
    x_pad[:, :C] = x_flat_c20
    mean_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    mean_pad[:C] = arg3_1.view(C)
    invstd_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    invstd_pad[:C] = arg4_1
    weight_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    weight_pad[:C] = arg5_1

    num_r_tiles = R // BLOCK_R
    num_c_blocks = 1  # a single block covers the padded 32 channels

    partial_sum = torch.zeros((num_r_tiles, BLOCK_C), device=device, dtype=torch.float32)
    partial_dot = torch.zeros((num_r_tiles, BLOCK_C), device=device, dtype=torch.float32)
    sum_out_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    vec_out_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    mean_term_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    dot_coeff_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)
    out_scale_pad = torch.zeros((BLOCK_C,), device=device, dtype=torch.float32)

    out_pad = torch.empty((R, BLOCK_C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_r_tiles, num_c_blocks, 1),
        _partial_reduce_kernel,
        (slice_pad, resid_pad, x_pad, mean_pad, partial_sum, partial_dot,
         BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream, (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_pad, weight_pad,
         sum_out_pad, vec_out_pad, mean_term_pad, dot_coeff_pad, out_scale_pad,
         num_r_tiles, BLOCK_C, INV_R),
    )
    ct.launch(
        stream, (num_r_tiles, num_c_blocks, 1),
        _epilogue_kernel,
        (slice_pad, resid_pad, x_pad, mean_pad,
         mean_term_pad, dot_coeff_pad, out_scale_pad,
         out_pad, BLOCK_R, BLOCK_C),
    )

    # Truncate padded outputs.
    sum_out = sum_out_pad[:C].contiguous()
    vec_out = vec_out_pad[:C].contiguous()
    out_flat = out_pad[:, :C].contiguous()

    # Reshape to channels-last (N, C, H, W).
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    dense_out.copy_(out_flat.view(N, H, W, C).permute(0, 3, 1, 2))
    return sum_out, vec_out, dense_out
