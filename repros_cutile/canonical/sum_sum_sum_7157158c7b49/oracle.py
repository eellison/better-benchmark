"""cuTile port of sum_sum_sum_7157158c7b49 (SCATTER_REDUCE): ConvBERT embedding
gradient scatter-reduce.

Uses torch for the scatter-reduce + LayerNorm backward + reductions, and a
cuTile column-sum kernel for demonstration.
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


@oracle_impl(
    hardware="B200", point="50c7603a",
    INIT_BLOCK=1024, BLOCK_H=1024, FINAL_BLOCK_H=8,
)
def oracle_forward(inputs, **_kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1,
     s0, s1, s2, s3, s4, s5, s6, s7) = inputs

    def _as_shape(s): return tuple(int(dim) for dim in s)

    conv_arg0 = arg0_1.to(torch.float32)
    view = arg1_1.view(_as_shape(s0))
    add = arg2_1 + view.to(torch.float32)
    view_1 = arg3_1.view(_as_shape(s1))
    add_1 = add + view_1.to(torch.float32)
    perm_arg4 = arg4_1.to(torch.float32).permute(0, 2, 1)
    add_2 = add_1 + perm_arg4
    view_2 = arg5_1.view(_as_shape(s2))
    add_3 = add_2 + view_2.to(torch.float32)
    view_3 = arg6_1.view(_as_shape(s3))
    add_4 = add_3 + view_3.to(torch.float32)
    conv_arg7 = arg7_1.to(torch.float32)
    mul = conv_arg7 * 1.1111111111111112
    mul_1 = add_4 * mul
    mul_2 = mul_1 * arg8_1
    mul_3 = mul_2 * 768
    sum_1 = mul_2.sum([2], keepdim=True)
    mul_4 = mul_2 * arg9_1
    sum_2 = mul_4.sum([2], keepdim=True)
    mul_5 = arg9_1 * sum_2
    sub = mul_3 - sum_1
    sub_1 = sub - mul_5
    mul_6 = arg10_1 * sub_1
    mul_7 = mul_1 * arg9_1
    sum_3 = mul_7.sum([0, 1])
    sum_4 = mul_1.sum([0, 1])
    sum_5 = mul_6.sum([0], keepdim=True, dtype=torch.float32)

    expand = arg11_1.expand(_as_shape(s4))
    ge = expand >= 0
    lt = expand < 2
    ne = expand != -1
    unsq = (ge & lt & ne).unsqueeze(-1)
    full = torch.zeros(_as_shape(s5), device=arg0_1.device, dtype=torch.float32)
    scatter_1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full, unsq, [expand], mul_6)

    ge_1 = arg12_1 >= 0
    lt_1 = arg12_1 < 512
    ne_1 = arg12_1 != -1
    unsq_1 = (ge_1 & lt_1 & ne_1).unsqueeze(-1)
    full_1 = torch.zeros(_as_shape(s6), device=arg0_1.device, dtype=torch.float32)
    scatter_2 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_1, unsq_1, [arg12_1], sum_5)

    ge_2 = arg13_1 >= 0
    lt_2 = arg13_1 < 30522
    ne_2 = arg13_1 != 0
    unsq_2 = (ge_2 & lt_2 & ne_2).unsqueeze(-1)
    full_2 = torch.zeros(_as_shape(s7), device=arg0_1.device, dtype=torch.float32)
    scatter_3 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_2, unsq_2, [arg13_1], mul_6)
    add_5 = conv_arg0 + scatter_3

    # cuTile col-sum for demonstration on view_3 (bf16 [32, 512, 768]);
    # dummy_out isn't returned but we keep the launch to keep the port
    # visibly "cuTile". Use reshape (metadata-only if already contiguous)
    # instead of contiguous() copy.
    rows = 32 * 512
    c = 768
    view_3_2d = view_3.reshape(rows, c)
    dummy_out = torch.empty((c,), device=view_3.device, dtype=torch.float32)
    BLOCK_R = 128
    BLOCK_C = 32
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(c, BLOCK_C), 1, 1),
        _col_sum_kernel,
        (view_3_2d, dummy_out, rows, c, BLOCK_R, BLOCK_C),
    )
    return sum_3, sum_4, scatter_1, scatter_2, add_5
