"""cuTile port of sum_sum_f967f297304c: DenseNet BN-backward tail.

Matches Triton's single-kernel structure: one @ct.kernel per channel that
does the sum reductions, dense bf16 gradient, and residual tail add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 736
SLICE_START = 704
SLICE_C = 32
SCALE = 0.0012755102040816326


@ct.kernel
def _densenet_tail_kernel(
    residual0_ptr,   # bf16 (N, 1024, HW)
    residual1_ptr,   # bf16 (N, 992, HW)
    residual2_ptr,   # bf16 (N, 960, HW)
    residual3_ptr,   # bf16 (N, 928, HW)
    residual4_ptr,   # bf16 (N, 896, HW)
    residual5_ptr,   # bf16 (N, 864, HW)
    residual6_ptr,   # bf16 (N, 832, HW)
    residual7_ptr,   # bf16 (N, 800, HW)
    residual8_ptr,   # bf16 (N, 768, HW)
    mask_ptr,        # bf16 (N, C=736, HW)
    fill_ptr,        # bf16 scalar
    source_ptr,      # bf16 (N, C=736, HW)
    centered_src_ptr, # bf16 (N, C=736, HW)
    mean_ptr,        # f32 (C,)
    invstd_ptr,      # f32 (C,)
    weight_ptr,      # f32 (C,)
    sum_out_ptr,     # f32 (C,)
    scaled_dot_out_ptr, # f32 (C,)
    dense_out_ptr,   # bf16 (N, C, HW)
    tail_out_ptr,    # bf16 (N, C, HW)  (over-allocated for grid simplicity)
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    c = ct.bid(0)

    # Load per-channel tile of shape (N, HW_PAD). Use 3D indexing on (N, C, HW).
    mask_tile = ct.load(mask_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                        padding_mode=ct.PaddingMode.ZERO)
    src_tile = ct.load(source_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    cen_tile = ct.load(centered_src_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))

    mask_f = ct.astype(ct.reshape(mask_tile, (N, HW_PAD)), ct.float32)
    src_f = ct.astype(ct.reshape(src_tile, (N, HW_PAD)), ct.float32)
    cen_f = ct.astype(ct.reshape(cen_tile, (N, HW_PAD)), ct.float32)
    fill_f = ct.astype(fill_tile, ct.float32)
    fill_scalar = ct.reshape(fill_f, (1, 1))
    fill_bcast = ct.broadcast_to(fill_scalar, (N, HW_PAD))

    source = ct.where(mask_f <= 0.0, fill_bcast, src_f)

    mean_v = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_f = ct.reshape(mean_v, (1, 1))
    mean_bcast = ct.broadcast_to(mean_f, (N, HW_PAD))
    centered = cen_f - mean_bcast

    # Mask out padding lanes so they don't pollute the reduction.
    hw_idx = ct.arange(HW_PAD, dtype=ct.int32)
    active_row = hw_idx < HW
    active_2d = ct.reshape(active_row, (1, HW_PAD))
    zero_2d = ct.zeros((N, HW_PAD), dtype=ct.float32)
    active_bcast = ct.broadcast_to(active_2d, (N, HW_PAD))
    source_m = ct.where(active_bcast, source, zero_2d)
    centered_m = ct.where(active_bcast, centered, zero_2d)

    sum_value = ct.sum(source_m, axis=None)  # scalar
    dot_value = ct.sum(source_m * centered_m, axis=None)  # scalar

    invstd_v = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_v = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd_f = ct.reshape(invstd_v, (1,))
    weight_f = ct.reshape(weight_v, (1,))

    sum_1d = ct.reshape(sum_value, (1,))
    dot_1d = ct.reshape(dot_value, (1,))
    mean_term = sum_1d * SCALE_C
    invstd_sq = invstd_f * invstd_f
    coeff = dot_1d * SCALE_C * invstd_sq
    output_scale = invstd_f * weight_f

    mean_term_2d = ct.reshape(mean_term, (1, 1))
    coeff_2d = ct.reshape(coeff, (1, 1))
    output_scale_2d = ct.reshape(output_scale, (1, 1))
    mean_term_bcast = ct.broadcast_to(mean_term_2d, (N, HW_PAD))
    coeff_bcast = ct.broadcast_to(coeff_2d, (N, HW_PAD))
    output_scale_bcast = ct.broadcast_to(output_scale_2d, (N, HW_PAD))

    correction = centered * coeff_bcast
    corrected = source - correction - mean_term_bcast
    dense_f = corrected * output_scale_bcast
    # bf16 round then back to f32 (matches Triton's _bf16_round_f32).
    dense_bf = ct.astype(dense_f, ct.bfloat16)
    dense_bf_f = ct.astype(dense_bf, ct.float32)

    # Store scalar per-channel outputs.
    ct.store(sum_out_ptr, index=(c,), tile=sum_1d)
    ct.store(scaled_dot_out_ptr, index=(c,), tile=dot_1d * invstd_f)
    # Store dense output tile — mask beyond HW is written but we don't care
    # since dense_out is allocated as (N, C, HW) exactly.
    dense_out_tile = ct.reshape(dense_bf, (N, 1, HW_PAD))
    ct.store(dense_out_ptr, index=(0, c, 0), tile=dense_out_tile)

    # Residual chain: only meaningful for c >= 704. We always compute + store,
    # then slice tail_out[:, 704:] outside. Loads reference position c in each
    # residual (which is valid since all have >= 768 channels).
    r0 = ct.load(residual0_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r1 = ct.load(residual1_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r2 = ct.load(residual2_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r3 = ct.load(residual3_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r4 = ct.load(residual4_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r5 = ct.load(residual5_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r6 = ct.load(residual6_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r7 = ct.load(residual7_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    r8 = ct.load(residual8_ptr, index=(0, c, 0), shape=(N, 1, HW_PAD),
                 padding_mode=ct.PaddingMode.ZERO)

    def _bf16_add(a, b):
        return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
                         ct.bfloat16)

    acc = _bf16_add(r0, r1)
    acc = _bf16_add(acc, r2)
    acc = _bf16_add(acc, r3)
    acc = _bf16_add(acc, r4)
    acc = _bf16_add(acc, r5)
    acc = _bf16_add(acc, r6)
    acc = _bf16_add(acc, r7)
    acc = _bf16_add(acc, r8)
    tail_value = _bf16_add(acc, dense_out_tile)
    ct.store(tail_out_ptr, index=(0, c, 0), tile=tail_value)


@oracle_impl(hardware="B200", point="1bdfd7ec", HW=196, HW_PAD=256)
@oracle_impl(hardware="B200", point="747f775b", HW=49, HW_PAD=64)
def oracle_forward(inputs, *, HW, HW_PAD):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        arg5_1, arg6_1, arg7_1, arg8_1, arg9_1,
        arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
    ) = inputs
    device = arg9_1.device
    height = int(arg9_1.shape[2])
    width = int(arg9_1.shape[3])
    hw = height * width
    assert hw == HW, f"{hw} != {HW}"

    # Slice each residual to (N, SLICE_C, H, W) and precompute the running add
    # in torch (matches Triton's residual chain via bf16 rounding).
    # Actually we do this inside the cuTile kernel above, so we just need the
    # source tensors with their native channel count.
    n = int(arg9_1.shape[0])
    assert n == N

    # Views (contiguous, NCHW).
    mask_v = arg9_1.view(N, C, hw)
    src_v = arg11_1.view(N, C, hw)
    cen_v = arg12_1.view(N, C, hw)
    mean_1d = arg13_1.view(C)
    fill_1d = arg10_1.view(1)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, height, width),
        (C * hw, hw, width, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_out_v = dense_out.view(N, C, hw)
    # Over-allocate tail_out to C channels; slice at the end.
    tail_full = torch.empty((N, C, hw), device=device, dtype=torch.bfloat16)

    r0_v = arg0_1.view(N, 1024, hw)
    r1_v = arg1_1.view(N, 992, hw)
    r2_v = arg2_1.view(N, 960, hw)
    r3_v = arg3_1.view(N, 928, hw)
    r4_v = arg4_1.view(N, 896, hw)
    r5_v = arg5_1.view(N, 864, hw)
    r6_v = arg6_1.view(N, 832, hw)
    r7_v = arg7_1.view(N, 800, hw)
    r8_v = arg8_1.view(N, 768, hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1),
        _densenet_tail_kernel,
        (r0_v, r1_v, r2_v, r3_v, r4_v, r5_v, r6_v, r7_v, r8_v,
         mask_v, fill_1d, src_v, cen_v,
         mean_1d, arg14_1, arg15_1,
         sum_out, scaled_dot_out, dense_out_v, tail_full,
         hw, HW_PAD, SCALE),
    )

    # Extract the true tail_out (c in [704, 736)).
    tail_slice = tail_full[:, SLICE_START:C].contiguous()
    tail_out = torch.empty_strided(
        (N, SLICE_C, height, width),
        (SLICE_C * hw, hw, width, 1),
        device=device, dtype=torch.bfloat16,
    )
    tail_out.copy_(tail_slice.view(N, SLICE_C, height, width))

    return sum_out, scaled_dot_out, dense_out, tail_out
