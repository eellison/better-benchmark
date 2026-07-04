"""cuTile port of sum_5a1cff122e9e: NFNet channels-last GELU-gradient + column sum.

Two-stage port mirroring Triton's split-K produce+finalize structure:
* _materialize_partial_kernel: bf16 add + gamma-scale (with bf16 rounding),
  in-kernel Abramowitz-Stegun erf polynomial + exp GELU-gradient, product,
  bf16 rounding, per-channel partial sum.
* _final_sum_kernel: sum partials across row groups, bf16-round to f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 128
H = 48
W = 48
HW = H * W
ROWS = N * HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@ct.kernel
def _materialize_partial_kernel(
    lhs0_ptr,        # bf16 [ROWS, C] contiguous (channels-last flattened)
    lhs1_ptr,        # bf16 [ROWS, C]
    rhs_ptr,         # bf16 [ROWS, C]
    out_ptr,         # bf16 [ROWS, C]
    partial_ptr,     # f32 [num_groups, C]
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    channel_group = ct.bid(1)

    lhs0 = ct.load(lhs0_ptr, index=(row_group, channel_group),
                   shape=(ROW_BLOCK, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    lhs1 = ct.load(lhs1_ptr, index=(row_group, channel_group),
                   shape=(ROW_BLOCK, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(rhs_ptr, index=(row_group, channel_group),
                  shape=(ROW_BLOCK, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)

    lhs0_f = ct.astype(lhs0, ct.float32)
    lhs1_f = ct.astype(lhs1, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)

    add_f32 = lhs0_f + lhs1_f
    add_bf16 = ct.astype(add_f32, ct.bfloat16)
    mul_one_bf16 = ct.astype(ct.astype(add_bf16, ct.float32), ct.bfloat16)
    scaled_for_sum = ct.astype(ct.astype(mul_one_bf16, ct.float32) * GAMMA,
                               ct.bfloat16)

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
    value_bf16 = ct.astype((add_f32 * GAMMA) * derivative, ct.bfloat16)
    value_for_sum_bf16 = ct.astype(ct.astype(scaled_for_sum, ct.float32) * derivative,
                                   ct.bfloat16)

    ct.store(out_ptr, index=(row_group, channel_group), tile=value_bf16)

    # ROW_BLOCK divides ROWS (294912) for the intended BLOCK sizes, and
    # BLOCK_C divides C (128), so no mask needed. Add mask defensively.
    row_base = row_group * ROW_BLOCK
    col_base = channel_group * BLOCK_C
    row_idx = ct.arange(ROW_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < ROWS, (ROW_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < C, (1, BLOCK_C))
    mask = row_valid & col_valid

    val_f = ct.astype(value_for_sum_bf16, ct.float32)
    masked = ct.where(mask, val_f, 0.0)
    partial = ct.sum(masked, axis=0)

    partial_2d = ct.reshape(partial, (1, BLOCK_C))
    ct.store(partial_ptr, index=(row_group, channel_group), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
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
    col_valid = ct.reshape((col_base + col_idx) < C, (1, BLOCK_C))
    mask = group_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(channel_group,), tile=rounded)


@oracle_impl(hardware="B200", point="8e74fca9", ROW_BLOCK=128, BLOCK_C=64, FINAL_BLOCK_C=32)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    lhs0, lhs1, rhs = inputs
    device = lhs0.device

    # lhs0/lhs1/rhs shape (N, C, H, W) with channels-last stride
    # (C*HW, 1, W*C, C) — so permute(0,2,3,1) produces contiguous (N, H, W, C)
    # which reshapes to (ROWS=N*HW, C). This matches Triton's offset
    # (n*294912 + h*6144 + w*128 + channels) mapping.
    lhs0_mc = lhs0.permute(0, 2, 3, 1).reshape(ROWS, C)
    lhs1_mc = lhs1.permute(0, 2, 3, 1).reshape(ROWS, C)
    rhs_mc = rhs.permute(0, 2, 3, 1).reshape(ROWS, C)

    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_mc = out.permute(0, 2, 3, 1).view(ROWS, C)

    num_groups = ct.cdiv(ROWS, ROW_BLOCK)
    partial = torch.empty((num_groups, C), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(C, BLOCK_C), 1),
        _materialize_partial_kernel,
        (lhs0_mc, lhs1_mc, rhs_mc, out_mc, partial, ROW_BLOCK, BLOCK_C),
    )
    ct.launch(
        stream,
        (ct.cdiv(C, FINAL_BLOCK_C), 1, 1),
        _final_sum_kernel,
        (partial, sum_out, num_groups, _ceil_pow2(num_groups), FINAL_BLOCK_C),
    )

    return out, sum_out
