"""cuTile port of sum_sum_sum_79c321089383: DistillGPT2 embedding-backward.

Fair port: row-wise LN backward reductions are computed inside `@ct.kernel`
via `ct.sum`. Full column reductions (`sum_3`, `sum_4`) are also computed
inside a cuTile kernel pair (partials + finalize) to mirror how Triton
computes them via `tl.atomic_add` inside its scatter kernel and the
`_finalize_hidden_kernel`. Scatter (position and vocab embeddings) stays in
torch since cuTile has no atomic scatter primitive matching Triton's exact
duplicate-index semantics for this dense output.

BLOCK_H = 1024 with `padding_mode=ct.PaddingMode.ZERO` to load valid columns
directly (mirrors Triton's masked tl.load) — no torch.zeros pad copy.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
BLOCK_H = 1024  # matches Triton's BLOCK_H=1024


@ct.kernel
def _row_reduce_kernel(
    mul_ptr,     # f32 (ROWS, HIDDEN)
    mul_5_ptr,   # f32 (ROWS, HIDDEN)
    sum_1_ptr,   # f32 (ROWS,)
    sum_2_ptr,   # f32 (ROWS,)
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    m = ct.load(mul_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                padding_mode=ct.PaddingMode.ZERO)
    m5 = ct.load(mul_5_ptr, index=(row, 0), shape=(1, BLOCK_H_),
                 padding_mode=ct.PaddingMode.ZERO)
    s1 = ct.sum(m)
    s2 = ct.sum(m5)
    ct.store(sum_1_ptr, index=(row,), tile=ct.reshape(s1, (1,)))
    ct.store(sum_2_ptr, index=(row,), tile=ct.reshape(s2, (1,)))


@ct.kernel
def _col_partials_kernel(
    view_ptr,    # f32 (ROWS, HIDDEN)
    mul_8_ptr,   # f32 (ROWS, HIDDEN)  == view * mul_4
    partials_ptr,  # f32 (NUM_ROW_BLOCKS, 2, HIDDEN)
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    row_block = ct.bid(1)
    v = ct.load(view_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS))
    m8 = ct.load(mul_8_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS))
    pv = ct.reshape(ct.sum(v, axis=0), (1, 1, BLOCK_COLS))
    p8 = ct.reshape(ct.sum(m8, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(row_block, 0, col_block), tile=p8)
    ct.store(partials_ptr, index=(row_block, 1, col_block), tile=pv)


@ct.kernel
def _col_finalize_kernel(
    partials_ptr,  # f32 (NUM_ROW_BLOCKS, 2, HIDDEN)
    sum_3_ptr,    # f32 (HIDDEN,)
    sum_4_ptr,    # f32 (HIDDEN,)
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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="e66c92a5",
             BLOCK_ROWS=64, BLOCK_COLS=64, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_COLS, FINAL_BLOCK_C):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1,
        shape_param_0, shape_param_1, shape_param_2, shape_param_3,
    ) = inputs
    device = arg0_1.device

    slice_1 = arg0_1[:50257, :]
    convert_element_type = slice_1.to(torch.float32)
    convert_element_type_1 = arg1_1.to(torch.float32)
    view = convert_element_type_1.view(_shape_tuple(shape_param_0))
    mul = view * arg2_1
    mul_1 = mul * 768
    add = arg3_1 + arg4_1
    mul_2 = arg5_1 * add
    mul_3 = mul_2 * 1.1111111111111112
    sub = mul_3 - arg6_1
    mul_4 = sub * arg7_1
    mul_5 = mul * mul_4

    # cuTile row-reduce kernel: sum_1 (mul.sum(dim=2)) and sum_2 (mul_5.sum(dim=2))
    # Directly view [ROWS, HIDDEN=768] — no pad copy; kernel loads with padding_mode=ZERO.
    mul_2d = mul.view(ROWS, HIDDEN)
    mul_5_2d = mul_5.view(ROWS, HIDDEN)
    sum_1_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    sum_2_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ROWS, 1, 1), _row_reduce_kernel,
              (mul_2d, mul_5_2d, sum_1_1d, sum_2_1d, HIDDEN, BLOCK_H))
    sum_1 = sum_1_1d.view(BATCH, SEQ, 1)
    sum_2 = sum_2_1d.view(BATCH, SEQ, 1)

    mul_6 = mul_4 * sum_2
    sub_1 = mul_1 - sum_1
    sub_2 = sub_1 - mul_6
    div = arg7_1 / 768
    mul_7 = div * sub_2

    # cuTile column-reduce kernels: sum_3 (mul_8.sum(dim=[0,1])), sum_4 (view.sum(dim=[0,1]))
    mul_8 = view * mul_4
    view_2d = view.view(ROWS, HIDDEN)
    mul_8_2d = mul_8.view(ROWS, HIDDEN)
    num_row_blocks = ROWS // BLOCK_ROWS
    partials = torch.empty((num_row_blocks, 2, HIDDEN), device=device, dtype=torch.float32)
    sum_3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    ct.launch(stream, (HIDDEN // BLOCK_COLS, num_row_blocks, 1),
              _col_partials_kernel,
              (view_2d, mul_8_2d, partials, BLOCK_ROWS, BLOCK_COLS))
    ct.launch(stream, (HIDDEN // FINAL_BLOCK_C, 1, 1),
              _col_finalize_kernel,
              (partials, sum_3, sum_4, num_row_blocks, FINAL_BLOCK_C))

    add_1 = arg8_1 + mul_7
    convert_element_type_2 = arg5_1.to(torch.float32)
    mul_9 = convert_element_type_2 * 1.1111111111111112
    mul_10 = add_1 * mul_9
    sum_5 = mul_10.sum(dim=0, keepdim=True, dtype=torch.float32)

    full = torch.full(_shape_tuple(shape_param_1), True, dtype=torch.bool, device=device)
    full_1 = torch.zeros(_shape_tuple(shape_param_2), device=device, dtype=torch.float32)
    scatter_pos = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_1, full, [arg9_1], sum_5,
    )

    ge = arg10_1 >= 0
    lt = arg10_1 < 50257
    bitwise_and = ge & lt
    ne = arg10_1 != -1
    bitwise_and_1 = bitwise_and & ne
    unsqueeze = bitwise_and_1.unsqueeze(-1)
    full_2 = torch.zeros(_shape_tuple(shape_param_3), device=device, dtype=torch.float32)
    scatter_vocab = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_2, unsqueeze, [arg10_1], mul_10,
    )
    add_2 = convert_element_type + scatter_vocab

    return sum_3, sum_4, scatter_pos, add_2
