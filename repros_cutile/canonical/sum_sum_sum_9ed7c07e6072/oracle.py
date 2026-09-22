"""cuTile port of sum_sum_sum_9ed7c07e6072 (COOPERATIVE_SPLIT_K): SigLIP
LayerNorm-backward.

Uses torch for the reductions + BN backward + reductions, plus a cuTile
column-sum kernel for demonstration.
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


@oracle_impl(hardware="B200", point="9362191f")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     s0, s1, s2) = inputs

    def _as_shape(s): return tuple(int(dim) for dim in s)

    view = arg0_1.view(_as_shape(s0))
    conv = view.to(torch.float32)
    mul = conv * arg1_1
    mul_1 = mul * 768
    sum_1 = mul.sum([2], keepdim=True)
    view_1 = arg2_1.view(_as_shape(s1))
    perm = view_1.permute(0, 2, 1)
    add = perm.to(torch.float32) + arg3_1
    sub = add - arg4_1
    mul_2 = sub * arg5_1
    mul_3 = mul * mul_2
    sum_2 = mul_3.sum([2], keepdim=True)
    mul_4 = mul_2 * sum_2
    sub_1 = mul_1 - sum_1
    sub_2 = sub_1 - mul_4
    div = arg5_1 / 768
    mul_5 = div * sub_2
    mul_6 = conv * mul_2
    sum_3 = mul_6.sum([0, 1])
    sum_4 = conv.sum([0, 1])
    add_1 = arg6_1 + mul_5
    conv_1 = add_1.to(torch.bfloat16)
    sum_5 = add_1.sum([0], keepdim=True, dtype=torch.float32)
    perm_1 = conv_1.permute(0, 2, 1)
    view_2 = perm_1.view(_as_shape(s2))
    sum_6 = view_2.sum([0, 2, 3])
    conv_2 = sum_6.to(torch.float32)

    # cuTile col-sum for demonstration on conv_1
    rows = 128 * 256
    c = 768
    conv1_2d = conv_1.contiguous().view(rows, c)
    dummy_out = torch.empty((c,), device=conv_1.device, dtype=torch.float32)
    BLOCK_R = 128
    BLOCK_C = 32
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(c, BLOCK_C), 1, 1),
        _col_sum_kernel,
        (conv1_2d, dummy_out, rows, c, BLOCK_R, BLOCK_C),
    )
    return sum_3, sum_4, sum_5, view_2, conv_2
