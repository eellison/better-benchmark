"""cuTile port of sum_sum_sum_74f116312f8b: DeBERTa embedding backward with scatter.

Fair port: row-wise LN backward reductions done in cuTile kernel; the full
column reductions (`sum_3`, `sum_4`) are done in a second cuTile kernel
(partials + finalize) to mirror how Triton computes them via `tl.atomic_add`
inside its `_seq_scatter_reduce_kernel`. Scatter (position and vocab) stays
in torch since cuTile has no atomic scatter primitive.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 512
HIDDEN = 1536
ROWS = BATCH * SEQ


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@ct.kernel
def _row_reduce_kernel(
    mul_ptr, mul4_ptr, sum_1_ptr, sum_2_ptr,
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    mul = ct.load(mul_ptr, index=(row, 0), shape=(1, BLOCK_H),
                  padding_mode=ct.PaddingMode.ZERO)
    mul4 = ct.load(mul4_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    mul_m = ct.where(col_mask, mul, zero)
    prod_m = ct.where(col_mask, mul * mul4, zero)
    s1 = ct.sum(mul_m)
    s2 = ct.sum(prod_m)
    ct.store(sum_1_ptr, index=(row,), tile=ct.reshape(s1, (1,)))
    ct.store(sum_2_ptr, index=(row,), tile=ct.reshape(s2, (1,)))


@ct.kernel
def _col_partials_kernel(
    mul_1_ptr,    # f32 (ROWS, HIDDEN)
    mul_8_ptr,    # f32 (ROWS, HIDDEN)
    partials_ptr, # f32 (NUM_ROW_BLOCKS, 2, HIDDEN)
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    row_block = ct.bid(1)
    m1 = ct.load(mul_1_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS))
    m8 = ct.load(mul_8_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS))
    p1 = ct.reshape(ct.sum(m1, axis=0), (1, 1, BLOCK_COLS))
    p8 = ct.reshape(ct.sum(m8, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(row_block, 0, col_block), tile=p8)
    ct.store(partials_ptr, index=(row_block, 1, col_block), tile=p1)


@ct.kernel
def _col_finalize_kernel(
    partials_ptr,  # f32 (NUM_ROW_BLOCKS, 2, HIDDEN)
    sum_3_ptr,     # f32 (HIDDEN,)
    sum_4_ptr,     # f32 (HIDDEN,)
    NUM_ROW_BLOCKS: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    p3 = ct.load(partials_ptr, index=(0, 0, col_block),
                 shape=(NUM_ROW_BLOCKS, 1, FINAL_BLOCK_C))
    p4 = ct.load(partials_ptr, index=(0, 1, col_block),
                 shape=(NUM_ROW_BLOCKS, 1, FINAL_BLOCK_C))
    s3 = ct.reshape(ct.sum(p3, axis=0), (FINAL_BLOCK_C,))
    s4 = ct.reshape(ct.sum(p4, axis=0), (FINAL_BLOCK_C,))
    ct.store(sum_3_ptr, index=(col_block,), tile=s3)
    ct.store(sum_4_ptr, index=(col_block,), tile=s4)


@oracle_impl(hardware="B200", point="40f378b3",
             BLOCK_ROWS=64, BLOCK_COLS=64, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_COLS, FINAL_BLOCK_C):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     arg9_1, arg10_1, arg11_1, arg12_1,
     shape0, shape1, shape2, shape3, shape4) = inputs
    device = arg0_1.device
    HIDDEN_ = int(arg1_1.shape[1])
    BLOCK_H = _next_pow2(HIDDEN_)

    slice_1 = arg0_1[:-4]
    convert = slice_1.float()

    view = arg1_1.view(8, 512, HIDDEN_).float()
    view_1 = arg3_1.view(8, 512, HIDDEN_).float()
    view_2 = arg4_1.view(8, 512, HIDDEN_).float()
    add_ = arg2_1 + view
    add_1 = add_ + view_1
    add_2 = add_1 + view_2
    mul_ = arg5_1.float() * 1.1111111111111112
    mul_1 = add_2 * mul_
    mul_2 = mul_1 * arg6_1
    mul_3 = mul_2 * 1536.0

    add_3 = arg7_1 + arg8_1
    sub_ = add_3 - arg9_1
    mul_4 = sub_ * arg10_1

    mul_2_2d = mul_2.view(ROWS, HIDDEN_)
    mul_4_2d = mul_4.view(ROWS, HIDDEN_)
    if BLOCK_H != HIDDEN_:
        m2p = torch.zeros((ROWS, BLOCK_H), device=device, dtype=torch.float32)
        m2p[:, :HIDDEN_].copy_(mul_2_2d)
        m4p = torch.zeros((ROWS, BLOCK_H), device=device, dtype=torch.float32)
        m4p[:, :HIDDEN_].copy_(mul_4_2d)
    else:
        m2p = mul_2_2d
        m4p = mul_4_2d
    sum_1_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    sum_2_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ROWS, 1, 1), _row_reduce_kernel,
              (m2p, m4p, sum_1_1d, sum_2_1d, HIDDEN_, BLOCK_H))
    sum_1 = sum_1_1d.view(8, 512, 1)
    sum_2 = sum_2_1d.view(8, 512, 1)

    mul_6 = mul_4 * sum_2
    sub_1 = mul_3 - sum_1
    sub_2 = sub_1 - mul_6
    div_ = arg10_1 / 1536.0
    mul_7 = div_ * sub_2

    # cuTile column-reduction kernel(s) for sum_3, sum_4.
    mul_8 = mul_1 * mul_4
    mul_1_2d = mul_1.view(ROWS, HIDDEN_)
    mul_8_2d = mul_8.view(ROWS, HIDDEN_)
    num_row_blocks = ROWS // BLOCK_ROWS
    partials = torch.empty((num_row_blocks, 2, HIDDEN_), device=device, dtype=torch.float32)
    sum_3 = torch.empty((HIDDEN_,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((HIDDEN_,), device=device, dtype=torch.float32)
    ct.launch(stream, (HIDDEN_ // BLOCK_COLS, num_row_blocks, 1),
              _col_partials_kernel,
              (mul_1_2d, mul_8_2d, partials, BLOCK_ROWS, BLOCK_COLS))
    ct.launch(stream, (HIDDEN_ // FINAL_BLOCK_C, 1, 1),
              _col_finalize_kernel,
              (partials, sum_3, sum_4, num_row_blocks, FINAL_BLOCK_C))

    sum_5 = mul_7.sum(dim=0, keepdim=True, dtype=torch.float32)
    ge_p = arg11_1 >= 0
    lt_p = arg11_1 < 512
    ne_p = arg11_1 != -1
    mask_p = (ge_p & lt_p & ne_p).unsqueeze(-1)
    scatter_pos = torch.zeros(tuple(int(d) for d in shape3), device=device, dtype=torch.float32)
    scatter_pos = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        scatter_pos, mask_p, [arg11_1], sum_5,
    )

    ge_w = arg12_1 >= 0
    lt_w = arg12_1 < 128100
    ne_w = arg12_1 != 0
    mask_w = (ge_w & lt_w & ne_w).unsqueeze(-1)
    scatter_word = torch.zeros(tuple(int(d) for d in shape4), device=device, dtype=torch.float32)
    scatter_word = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        scatter_word, mask_w, [arg12_1], mul_7,
    )
    add_4 = convert + scatter_word

    return sum_3, sum_4, scatter_pos, add_4
