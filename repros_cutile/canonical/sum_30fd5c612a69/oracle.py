"""cuTile port of sum_30fd5c612a69: NFNet crop-gather GELU derivative + channel sum.

Two-stage port mirroring Triton's split-K produce+finalize structure:
* _materialize_partial_kernel: fused pointwise (in-kernel Abramowitz-Stegun erf
  polynomial, exp, bf16 rounding boundaries) + per-channel partial sum.
* _final_sum_kernel: sum partials across row groups, round through bf16.

The crop (`constant_pad_nd`-style top-left crop) is done as a strided torch
view + contiguous pre-pass; this same crop is unavoidable in cuTile since we
lack pointer-arithmetic style gather loads and the source has a channels-last
stride mismatch. Triton fuses the crop via computed offsets.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


@ct.kernel
def _materialize_partial_kernel(
    lhs_ptr,        # bf16 [M, C] contiguous (channels-last flattened)
    rhs_ptr,        # bf16 [M, C] contiguous
    out_ptr,        # bf16 [M, C] contiguous (view of channels-last out)
    partial_ptr,    # f32 [num_groups, C]
    sum_ptr,        # f32 [C]
    M_: ct.Constant[int],
    C_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    STORE_DIRECT_SUM: ct.Constant[bool],
):
    row_group = ct.bid(0)
    channel_group = ct.bid(1)

    lhs = ct.load(lhs_ptr, index=(row_group, channel_group),
                  shape=(ROW_BLOCK, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(rhs_ptr, index=(row_group, channel_group),
                  shape=(ROW_BLOCK, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)

    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)

    scaled_lhs = lhs_f * GAMMA
    scaled_lhs_for_sum = ct.astype(ct.astype(scaled_lhs, ct.bfloat16), ct.float32)

    # In-kernel erf via Abramowitz-Stegun 7.1.26 polynomial approximation.
    erf_arg = rhs_f * RSQRT2
    zero_f = ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((ROW_BLOCK, BLOCK_C), 1.0, dtype=ct.float32),
                    ct.full((ROW_BLOCK, BLOCK_C), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_val = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    cdf = (erf_val + 1.0) * 0.5
    pdf_term = rhs_f * (ct.exp((rhs_f * rhs_f) * -0.5) * NORMAL_PDF_SCALE)
    derivative = cdf + pdf_term
    value_bf16 = ct.astype(scaled_lhs * derivative, ct.bfloat16)
    value_for_sum_bf16 = ct.astype(scaled_lhs_for_sum * derivative, ct.bfloat16)

    ct.store(out_ptr, index=(row_group, channel_group), tile=value_bf16)

    # Mask for OOB rows (channels always fit because we assume BLOCK_C divides C).
    row_base = row_group * ROW_BLOCK
    col_base = channel_group * BLOCK_C
    row_idx = ct.arange(ROW_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < M_, (ROW_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = row_valid & col_valid

    value_for_sum_f = ct.astype(value_for_sum_bf16, ct.float32)
    masked = ct.where(mask, value_for_sum_f, 0.0)
    partial = ct.sum(masked, axis=0)  # (BLOCK_C,)

    if STORE_DIRECT_SUM:
        rounded = ct.astype(ct.astype(partial, ct.bfloat16), ct.float32)
        ct.store(sum_ptr, index=(channel_group,), tile=rounded)
    else:
        partial_2d = ct.reshape(partial, (1, BLOCK_C))
        ct.store(partial_ptr, index=(row_group, channel_group), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    C_: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    channel_group = ct.bid(0)

    values = ct.load(partial_ptr, index=(0, channel_group),
                     shape=(GROUP_BLOCK, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    values_f = ct.astype(values, ct.float32)

    col_base = channel_group * BLOCK_C
    group_idx = ct.arange(GROUP_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    group_valid = ct.reshape(group_idx < NUM_GROUPS, (GROUP_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = group_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(channel_group,), tile=rounded)


@oracle_impl(hardware="B200", point="c8e8c8cf", ROW_BLOCK=128, BLOCK_C=64, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="b006f82e", ROW_BLOCK=128, BLOCK_C=32, FINAL_BLOCK_C=32)
@oracle_impl(hardware="B200", point="b37b1da0", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32)
@oracle_impl(hardware="B200", point="4f4e0306", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    crop_source, rhs = inputs
    device = rhs.device
    N, C, H, W = (int(dim) for dim in rhs.shape)
    M = N * H * W

    # Materialize the crop and rearrange to a (M, C) contiguous view.
    # Both rhs and the target `out` are channels-last (stride[C]=1, stride[W]=C,
    # stride[H]=W*C, stride[N]=big), so permute(0,2,3,1) is contiguous and
    # can be reshaped to (M, C). The cropped `crop_source[:, :, :H, :W]` is a
    # non-contiguous strided view (spatial stride differs) and requires copy.
    lhs = crop_source[:, :, :H, :W].permute(0, 2, 3, 1).contiguous().view(M, C)
    rhs_mc = rhs.permute(0, 2, 3, 1).reshape(M, C)

    out = torch.empty_strided(
        (N, C, H, W), tuple(int(s) for s in rhs.stride()),
        device=device, dtype=torch.bfloat16,
    )
    out_mc = out.permute(0, 2, 3, 1).view(M, C)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)

    num_groups = ct.cdiv(M, ROW_BLOCK)
    direct_sum = num_groups == 1
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty_strided(
            (num_groups, C), (C, 1),
            device=device, dtype=torch.float32,
        )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(C, BLOCK_C), 1),
        _materialize_partial_kernel,
        (lhs, rhs_mc, out_mc, partial, sum_out,
         M, C, ROW_BLOCK, BLOCK_C, direct_sum),
    )

    if not direct_sum:
        group_block = 1 << (num_groups - 1).bit_length()
        ct.launch(
            stream,
            (ct.cdiv(C, FINAL_BLOCK_C), 1, 1),
            _final_sum_kernel,
            (partial, sum_out, C, num_groups, group_block, FINAL_BLOCK_C),
        )

    return out, sum_out
