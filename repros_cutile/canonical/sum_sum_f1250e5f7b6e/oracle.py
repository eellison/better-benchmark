"""cuTile port of sum_sum_f1250e5f7b6e: T5 residual/dropout/LN backward.

Two cuTile kernels: (1) per-row producer that computes total = base + 11 residuals,
dropped = total*mask1*MASK_SCALE, weighted = dropped*weight, weighted*src row-sum,
and dropped*src*row_scale col-sum (atomic add); (2) per-row epilogue that
uses row_total from step 1 to produce dense_f32 and dense_bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 512
MASK_SCALE = 1.1111111111111112
INV_COLS = 0.001953125  # 1/512


@ct.kernel
def _row_producer_kernel(
    base_ptr,        # bf16 (ROWS, COLS)
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr,
    r6_ptr, r7_ptr, r8_ptr, r9_ptr, r10_ptr, r11_ptr,
    mask1_ptr,       # bf16 (ROWS, COLS)
    weight_ptr,      # bf16 (COLS,)
    src_ptr,         # f32 (ROWS, COLS)
    row_scale_ptr,   # f32 (ROWS,)
    weighted_ptr,    # f32 (ROWS, COLS)
    row_total_ptr,   # f32 (ROWS,)
    col_sum_ptr,     # f32 (COLS,) — atomic add
    COLS_C: ct.Constant[int],
):
    row = ct.bid(0)

    base = ct.astype(ct.load(base_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r0 = ct.astype(ct.load(r0_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r1 = ct.astype(ct.load(r1_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r2 = ct.astype(ct.load(r2_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r3 = ct.astype(ct.load(r3_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r4 = ct.astype(ct.load(r4_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r5 = ct.astype(ct.load(r5_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r6 = ct.astype(ct.load(r6_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r7 = ct.astype(ct.load(r7_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r8 = ct.astype(ct.load(r8_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r9 = ct.astype(ct.load(r9_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r10 = ct.astype(ct.load(r10_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    r11 = ct.astype(ct.load(r11_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)

    total = base + r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11
    mask1 = ct.astype(ct.load(mask1_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    dropped = total * (mask1 * MASK_SCALE)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(COLS_C,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, COLS_C))
    weighted = dropped * weight_2d
    ct.store(weighted_ptr, index=(row, 0), tile=weighted)

    src = ct.load(src_ptr, index=(row, 0), shape=(1, COLS_C))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    row_terms = weighted * src
    row_sum_scalar = ct.sum(row_terms)
    ct.store(row_total_ptr, index=(row,), tile=ct.reshape(row_sum_scalar, (1,)))

    src_scale = src * row_scale_2d
    col_terms = dropped * src_scale
    cols = ct.arange(COLS_C, dtype=ct.int32)
    col_terms_1d = ct.reshape(col_terms, (COLS_C,))
    ct.atomic_add(col_sum_ptr, (cols,), col_terms_1d)


@ct.kernel
def _epilogue_kernel(
    weighted_ptr,   # f32 (ROWS, COLS)
    src_ptr,        # f32 (ROWS, COLS)
    row_scale_ptr,  # f32 (ROWS,)
    mask2_ptr,      # bf16 (ROWS, COLS)
    row_total_ptr,  # f32 (ROWS,)
    dense_f32_ptr,  # f32 (ROWS, COLS)
    dense_bf16_ptr, # bf16 (ROWS, COLS)
    COLS_C: ct.Constant[int],
    INV_COLS_C: ct.Constant[float],
):
    row = ct.bid(0)

    weighted = ct.load(weighted_ptr, index=(row, 0), shape=(1, COLS_C))
    src = ct.load(src_ptr, index=(row, 0), shape=(1, COLS_C))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    row_total = ct.load(row_total_ptr, index=(row,), shape=(1,))

    direct = weighted * row_scale_2d
    row_scale_sq = row_scale * row_scale
    row_scale_cu = row_scale_sq * row_scale
    correction0 = row_total * -0.5
    correction1 = correction0 * row_scale_cu
    correction2 = correction1 * INV_COLS_C
    correction2_2d = ct.reshape(correction2, (1, 1))
    correction3 = src * 2.0
    correction = correction2_2d * correction3
    dense_f32 = direct + correction

    mask2 = ct.astype(ct.load(mask2_ptr, index=(row, 0), shape=(1, COLS_C)), ct.float32)
    dense_bf16 = dense_f32 * (mask2 * MASK_SCALE)

    ct.store(dense_f32_ptr, index=(row, 0), tile=dense_f32)
    ct.store(dense_bf16_ptr, index=(row, 0), tile=ct.astype(dense_bf16, ct.bfloat16))


@oracle_impl(hardware="B200", point="6927d79e")
def oracle_forward(inputs):
    (
        residual0, base, residual1, residual2, residual3, residual4,
        residual5, residual6, residual7, residual8, residual9,
        residual10, residual11,
        mask1, weight, src, row_scale, mask2,
        *_shape_params,
    ) = inputs
    device = base.device

    row_total = torch.zeros((ROWS,), device=device, dtype=torch.float32)
    weighted = torch.empty_strided((ROWS, COLS), (COLS, 1),
                                   device=device, dtype=torch.float32)
    col_sum = torch.zeros((COLS,), device=device, dtype=torch.float32)
    dense_f32 = torch.empty_strided((8, 1024, COLS), (1024 * COLS, COLS, 1),
                                    device=device, dtype=torch.float32)
    dense_bf16 = torch.empty_strided((ROWS, COLS), (COLS, 1),
                                     device=device, dtype=torch.bfloat16)

    dense_f32_2d = dense_f32.view(ROWS, COLS)

    # Base and mask2 are 3D; reshape to 2D matching ROWS x COLS.
    base_2d = base.reshape(ROWS, COLS)
    mask1_2d = mask1.reshape(ROWS, COLS)
    src_2d = src.reshape(ROWS, COLS)
    row_scale_1d = row_scale.reshape(ROWS)
    mask2_2d = mask2.reshape(ROWS, COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _row_producer_kernel,
        (
            base_2d, residual0, residual1, residual2, residual3, residual4,
            residual5, residual6, residual7, residual8, residual9,
            residual10, residual11,
            mask1_2d, weight, src_2d, row_scale_1d,
            weighted, row_total, col_sum, COLS,
        ),
    )
    ct.launch(
        stream, (ROWS, 1, 1), _epilogue_kernel,
        (
            weighted, src_2d, row_scale_1d, mask2_2d, row_total,
            dense_f32_2d, dense_bf16,
            COLS, INV_COLS,
        ),
    )
    return col_sum, dense_f32, dense_bf16, dense_bf16.permute(1, 0)
