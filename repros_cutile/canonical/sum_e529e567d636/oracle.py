"""cuTile port of sum_e529e567d636: BERT/XLNet dropout-scaled GELU-derivative.

Matches Triton's structure: one kernel does dropout-scaled lhs, erf-based
GELU derivative (via Abramowitz-Stegun 7.1.26 approx), materialization, and
partial column-sum. If num_groups==1 it stores the sum directly; otherwise a
second kernel finalizes the partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _materialize_partial_kernel(
    lhs_ptr,          # bf16 [M, N]
    keep_ptr,         # b8 [M, N] (viewed same layout)
    rhs_ptr,          # bf16 [M, N]
    out_ptr,          # bf16 [M, N]
    partial_ptr,      # f32 [NUM_GROUPS, N] or same as sum_ptr when direct
    sum_ptr,          # f32 [N]
    M_: ct.Constant[int],
    N_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    STORE_DIRECT_SUM: ct.Constant[bool],
):
    row_group = ct.bid(0)
    col_group = ct.bid(1)

    lhs = ct.load(lhs_ptr, index=(row_group, col_group),
                  shape=(ROW_BLOCK, BLOCK_N),
                  padding_mode=ct.PaddingMode.ZERO)
    keep = ct.load(keep_ptr, index=(row_group, col_group),
                   shape=(ROW_BLOCK, BLOCK_N),
                   padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(rhs_ptr, index=(row_group, col_group),
                  shape=(ROW_BLOCK, BLOCK_N),
                  padding_mode=ct.PaddingMode.ZERO)

    lhs_f = ct.astype(lhs, ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)

    # Dropout scale: keep_scale = bf16(keep * 1.11111...); lhs_scaled = bf16(lhs * keep_scale)
    keep_scale_bf = ct.astype(keep_f * 1.1111111111111112, ct.bfloat16)
    keep_scale_f = ct.astype(keep_scale_bf, ct.float32)
    lhs_scaled_bf = ct.astype(lhs_f * keep_scale_f, ct.bfloat16)
    lhs_scaled_f = ct.astype(lhs_scaled_bf, ct.float32)

    # In-kernel erf via Abramowitz-Stegun 7.1.26.
    erf_arg = rhs_f * 0.7071067811865476
    zero_f = ct.zeros((ROW_BLOCK, BLOCK_N), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((ROW_BLOCK, BLOCK_N), 1.0, dtype=ct.float32),
                    ct.full((ROW_BLOCK, BLOCK_N), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_f = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    cdf = (erf_f + 1.0) * 0.5
    exp_arg = (rhs_f * rhs_f) * (-0.5)
    pdf_term = rhs_f * (ct.exp(exp_arg) * 0.3989422804014327)
    value = lhs_scaled_f * (cdf + pdf_term)
    value_bf16 = ct.astype(value, ct.bfloat16)

    ct.store(out_ptr, index=(row_group, col_group), tile=value_bf16)

    # Row-mask for validity
    row_base = row_group * ROW_BLOCK
    col_base = col_group * BLOCK_N
    row_idx = ct.arange(ROW_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < M_, (ROW_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < N_, (1, BLOCK_N))
    mask = row_valid & col_valid

    value_f = ct.astype(value_bf16, ct.float32)
    masked_val = ct.where(mask, value_f, 0.0)
    partial = ct.sum(masked_val, axis=0)  # (BLOCK_N,)

    if STORE_DIRECT_SUM:
        rounded = ct.astype(ct.astype(partial, ct.bfloat16), ct.float32)
        ct.store(sum_ptr, index=(col_group,), tile=rounded)
    else:
        partial_2d = ct.reshape(partial, (1, BLOCK_N))
        ct.store(partial_ptr, index=(row_group, col_group), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,      # f32 [NUM_GROUPS, N]
    sum_ptr,          # f32 [N]
    N_: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_group = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, col_group),
                     shape=(GROUP_BLOCK, BLOCK_N),
                     padding_mode=ct.PaddingMode.ZERO)
    values_f = ct.astype(values, ct.float32)

    col_base = col_group * BLOCK_N
    group_idx = ct.arange(GROUP_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    group_valid = ct.reshape(group_idx < NUM_GROUPS, (GROUP_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < N_, (1, BLOCK_N))
    mask = group_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_group,), tile=rounded)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="ed3cce87", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16)
@oracle_impl(hardware="B200", point="fea06368", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_N: int, FINAL_BLOCK_N: int):
    lhs, keep, rhs, _shape0, _shape1, shape2, shape3 = inputs
    m, n = _shape_tuple(shape2)
    device = lhs.device

    lhs2 = lhs.view(m, n)
    keep2 = keep.view(m, n)  # b8, same layout
    rhs2 = rhs.view(m, n)

    out = torch.empty_strided((m, n), (n, 1), device=device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(_shape_tuple(shape3), (1,), device=device, dtype=torch.float32)

    num_groups = ct.cdiv(m, ROW_BLOCK)
    direct_sum = num_groups == 1
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty_strided(
            (num_groups, n),
            (n, 1),
            device=device,
            dtype=torch.float32,
        )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(n, BLOCK_N), 1),
        _materialize_partial_kernel,
        (lhs2, keep2, rhs2, out, partial, sum_out,
         m, n, ROW_BLOCK, BLOCK_N, direct_sum),
    )

    if not direct_sum:
        group_block = 1 << (num_groups - 1).bit_length()
        ct.launch(
            stream,
            (ct.cdiv(n, FINAL_BLOCK_N), 1, 1),
            _final_sum_kernel,
            (partial, sum_out, n, num_groups, group_block, FINAL_BLOCK_N),
        )

    return out, out.permute(1, 0), sum_out
