"""cuTile port of sum_sum_sum_301d492a313d (SCATTER_REDUCE): U-Net max-pool
backward + BN backward + reductions.

Uses torch for the aten scatter_add + BN backward reductions, then a cuTile
column-sum kernel for the final channel reduction.
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


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(
    hardware="B200", point="ca50fc2e",
    REDUCE_BLOCK_HW=256, REDUCE_BLOCK_C=8, FINAL_BLOCK_C=8,
    EPILOGUE_BLOCK_HW=256, EPILOGUE_BLOCK_C=8, OUT_SUM_BLOCK_C=8,
    reduce_warps=8, finalize_warps=8, epilogue_warps=8, out_sum_warps=8,
)
def oracle_forward(inputs, **_kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     s0, s1, s2, s3, s4, s5, s6) = inputs

    def _as_shape(s): return tuple(int(dim) for dim in s)
    s0, s1, s2, s3, s4, s5, s6 = (_as_shape(s) for s in (s0, s1, s2, s3, s4, s5, s6))

    slice_1 = arg0_1[:, 0:128]
    full = torch.zeros(s0, device=arg0_1.device, dtype=torch.float32)
    view = arg1_1.view(s1)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg2_1, s2, s3, s4, [0, 0], [1, 1])
    view_1 = indices.view(s5)
    convert = view.to(torch.float32)
    scattered = torch.scatter_add(full, 1, view_1, convert)
    view_2 = scattered.view(s6)
    conv_bf = view_2.to(torch.bfloat16)
    add = slice_1 + conv_bf

    sub = arg3_1 - arg4_1
    mul = sub * arg5_1
    mul_1 = mul * arg6_1.view(1, 128, 1, 1)
    add_1 = mul_1 + arg7_1.view(1, 128, 1, 1)
    conv_bf1 = add_1.to(torch.bfloat16)
    relu_val = torch.relu(conv_bf1)
    le = relu_val <= 0
    where_out = torch.where(le, arg8_1, add)
    where_f32 = where_out.to(torch.float32)

    squeeze = arg4_1.squeeze([0, 2, 3])
    mean_view = squeeze.view(1, 128, 1, 1)
    sum_1 = where_f32.sum([0, 2, 3])

    conv_3f = arg3_1.to(torch.float32)
    sub_1 = conv_3f - mean_view
    mul_2 = where_f32 * sub_1
    sum_2 = mul_2.sum([0, 2, 3])
    mul_3 = sum_1 * 6.524008350730689e-06
    mul_4 = sum_2 * 6.524008350730689e-06

    squeeze_1 = arg5_1.squeeze([0, 2, 3])
    mul_5 = squeeze_1 * squeeze_1
    mul_6 = mul_4 * mul_5
    mul_7 = squeeze_1 * arg6_1
    u12 = mul_6.view(1, 128, 1, 1)
    u15 = mul_7.view(1, 128, 1, 1)
    u9 = mul_3.view(1, 128, 1, 1)
    mul_8 = sub_1 * u12
    sub_2 = where_f32 - mul_8
    sub_3 = sub_2 - u9
    mul_9 = sub_3 * u15
    mul_10 = sum_2 * squeeze_1
    conv_5 = mul_9.to(torch.bfloat16)

    # sum_3 via cuTile
    # conv_5 shape [1, 128, 320, 479]. Rows = 1*320*479, C = 128.
    n, c, h, w = conv_5.shape
    rows = n * h * w  # 1 * 320 * 479 = 153280
    conv_5_2d = conv_5.permute(0, 2, 3, 1).contiguous().view(rows, c)
    sum_3_f32 = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    BLOCK_R = 128
    BLOCK_C = 128
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(c, BLOCK_C), 1, 1),
        _col_sum_kernel,
        (conv_5_2d, sum_3_f32, rows, c, BLOCK_R, BLOCK_C),
    )

    return sum_1, mul_10, conv_5, sum_3_f32
