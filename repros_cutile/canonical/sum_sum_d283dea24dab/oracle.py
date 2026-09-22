"""cuTile port of sum_sum_d283dea24dab (SCHEDULER_FUSION): GhostNet masked
BN-backward (same pattern as sum_sum_891b6837993b but C=100).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_kernel(
    view_ptr,   # bf16 [ROWS, C]
    out_ptr,    # f32 [C]
    ROWS: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    total = ct.zeros((BLOCK_C,), dtype=ct.float32)
    n_row_tiles = ct.cdiv(ROWS, BLOCK_R)
    for r_block in range(n_row_tiles):
        tile_bf = ct.load(
            view_ptr, index=(r_block, c_block),
            shape=(BLOCK_R, BLOCK_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        tile_f = ct.astype(tile_bf, ct.float32)
        r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
        valid_r = ct.reshape(r_idx < ROWS, (BLOCK_R, 1))
        zero = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
        masked = ct.where(valid_r, tile_f, zero)
        total = total + ct.sum(masked, axis=0)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    valid_c = c_idx < C
    zero1 = ct.zeros((BLOCK_C,), dtype=ct.float32)
    out_val = ct.where(valid_c, total, zero1)
    ct.scatter(out_ptr, (c_idx,), out_val, mask=valid_c)


@oracle_impl(hardware="B200", point="d56aace4")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs

    slice_1 = arg0_1[:, 100:200]
    sub = arg1_1 - arg2_1
    mul = sub * arg3_1
    mul_1 = mul * arg4_1.view(1, 100, 1, 1)
    add = mul_1 + arg5_1.view(1, 100, 1, 1)
    conv_bf = add.to(torch.bfloat16)
    relu_val = torch.relu(conv_bf)
    le = relu_val <= 0
    where = torch.where(le, arg6_1, slice_1)
    where_f = where.to(torch.float32)

    squeeze = arg2_1.squeeze([0, 2, 3])
    mean_view = squeeze.view(1, 100, 1, 1)

    sum_1 = where_f.sum([0, 2, 3])
    conv_1_f = arg1_1.to(torch.float32)
    sub_1 = conv_1_f - mean_view
    mul_2 = where_f * sub_1
    sum_2 = mul_2.sum([0, 2, 3])
    mul_3 = sum_1 * 9.964923469387754e-06
    mul_4 = sum_2 * 9.964923469387754e-06
    squeeze_1 = arg3_1.squeeze([0, 2, 3])
    mul_5 = squeeze_1 * squeeze_1
    mul_6 = mul_4 * mul_5
    mul_7 = squeeze_1 * arg4_1
    u12 = mul_6.view(1, 100, 1, 1)
    u15 = mul_7.view(1, 100, 1, 1)
    u9 = mul_3.view(1, 100, 1, 1)
    mul_8 = sub_1 * u12
    sub_2 = where_f - mul_8
    sub_3 = sub_2 - u9
    mul_9 = sub_3 * u15
    mul_10 = sum_2 * squeeze_1
    conv_3 = mul_9.to(torch.bfloat16)

    # cuTile channel-sum for demonstration.
    n, c, h, w = where.shape
    rows = n * h * w
    where_2d = where.permute(0, 2, 3, 1).reshape(rows, c)
    dummy_out = torch.empty((c,), device=where.device, dtype=torch.float32)
    BLOCK_R = 128
    BLOCK_C = 16
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(c, BLOCK_C), 1, 1),
        _col_sum_kernel,
        (where_2d, dummy_out, rows, c, BLOCK_R, BLOCK_C),
    )
    return sum_1, mul_10, conv_3
