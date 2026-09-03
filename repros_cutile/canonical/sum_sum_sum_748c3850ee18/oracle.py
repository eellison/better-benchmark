"""cuTile port of sum_sum_sum_748c3850ee18: ALBERT column reduction + layout.

Fair port: 2 kernels matching Triton structure. Kernel 1 computes per-row-block
partial column sums for 11 matrices AND materializes the layout-permuted+scaled
clone (writing its partial column sums too). Kernel 2 finalizes 12 partials with
bf16 rounding boundaries and an f32 add chain.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
CHANNELS = 4096
SCALE = 0.3535533905932738


@ct.kernel
def _all_partials_and_layout_kernel(
    m0_ptr, m1_ptr, m2_ptr, m3_ptr, m4_ptr, m5_ptr,
    m6_ptr, m7_ptr, m8_ptr, m9_ptr, m10_ptr,
    layout_in_ptr,   # bf16 flat (2097152 * 8,)
    clone_out_ptr,   # bf16 (ROWS, CHANNELS)
    partials_ptr,    # f32 (12, NUM_ROW_BLOCKS, CHANNELS)
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    row_block = ct.bid(1)

    # 11 input matrices: partial column sums via ct.sum(axis=0)
    t0 = ct.astype(ct.load(m0_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p0 = ct.reshape(ct.sum(t0, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(0, row_block, col_block), tile=p0)

    t1 = ct.astype(ct.load(m1_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p1 = ct.reshape(ct.sum(t1, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(1, row_block, col_block), tile=p1)

    t2 = ct.astype(ct.load(m2_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p2 = ct.reshape(ct.sum(t2, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(2, row_block, col_block), tile=p2)

    t3 = ct.astype(ct.load(m3_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p3 = ct.reshape(ct.sum(t3, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(3, row_block, col_block), tile=p3)

    t4 = ct.astype(ct.load(m4_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p4 = ct.reshape(ct.sum(t4, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(4, row_block, col_block), tile=p4)

    t5 = ct.astype(ct.load(m5_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p5 = ct.reshape(ct.sum(t5, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(5, row_block, col_block), tile=p5)

    t6 = ct.astype(ct.load(m6_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p6 = ct.reshape(ct.sum(t6, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(6, row_block, col_block), tile=p6)

    t7 = ct.astype(ct.load(m7_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p7 = ct.reshape(ct.sum(t7, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(7, row_block, col_block), tile=p7)

    t8 = ct.astype(ct.load(m8_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p8 = ct.reshape(ct.sum(t8, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(8, row_block, col_block), tile=p8)

    t9 = ct.astype(ct.load(m9_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p9 = ct.reshape(ct.sum(t9, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(9, row_block, col_block), tile=p9)

    t10 = ct.astype(ct.load(m10_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS)), ct.float32)
    p10 = ct.reshape(ct.sum(t10, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(10, row_block, col_block), tile=p10)

    # Layout matrix: gather-permute + scale + bf16 rounding + store clone + partial column sum
    rows_1d = ct.arange(BLOCK_ROWS, dtype=ct.int32) + row_block * BLOCK_ROWS
    cols_1d = ct.arange(BLOCK_COLS, dtype=ct.int32) + col_block * BLOCK_COLS
    batch = ct.reshape(rows_1d // 512, (BLOCK_ROWS, 1))
    seq = ct.reshape(rows_1d - (rows_1d // 512) * 512, (BLOCK_ROWS, 1))
    head = ct.reshape(cols_1d // 64, (1, BLOCK_COLS))
    dim = ct.reshape(cols_1d - (cols_1d // 64) * 64, (1, BLOCK_COLS))
    off = batch * (64 * 64 * 512) + head * (64 * 512) + dim * 512 + seq
    values = ct.gather(layout_in_ptr, off)
    values_f = ct.astype(values, ct.float32)
    scaled_bf = ct.astype(values_f * SCALE, ct.bfloat16)
    ct.store(clone_out_ptr, index=(row_block, col_block), tile=scaled_bf)
    scaled_f = ct.astype(scaled_bf, ct.float32)
    p11 = ct.reshape(ct.sum(scaled_f, axis=0), (1, 1, BLOCK_COLS))
    ct.store(partials_ptr, index=(11, row_block, col_block), tile=p11)


@ct.kernel
def _finalize_kernel(
    partials_ptr,   # f32 (12, NUM_ROW_BLOCKS, CHANNELS)
    out_ptr,        # f32 (CHANNELS,)
    NUM_ROW_BLOCKS: ct.Constant[int],
    FINAL_BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)

    t = ct.load(partials_ptr, index=(0, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(1, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(2, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(3, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(4, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(5, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(6, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(7, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(8, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(9, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(10, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    t = ct.load(partials_ptr, index=(11, 0, col_block), shape=(1, NUM_ROW_BLOCKS, FINAL_BLOCK_COLS))
    s = ct.reshape(ct.sum(t, axis=1), (FINAL_BLOCK_COLS,))
    total = total + ct.astype(ct.astype(s, ct.bfloat16), ct.float32)

    ct.store(out_ptr, index=(col_block,), tile=total)


@oracle_impl(
    hardware="B200",
    point="f2a6d7d1",
    BLOCK_ROWS=128,
    BLOCK_COLS=64,
    FINAL_BLOCK_COLS=16,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_COLS: int,
    FINAL_BLOCK_COLS: int,
):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
        *_shape_args,
    ) = inputs

    num_row_blocks = ROWS // BLOCK_ROWS
    device = arg0.device

    clone = torch.empty_strided(
        (ROWS, CHANNELS), (CHANNELS, 1), device=device, dtype=torch.bfloat16
    )
    partials = torch.empty(
        (12, num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    src_flat = arg11.contiguous().view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (CHANNELS // BLOCK_COLS, num_row_blocks, 1),
        _all_partials_and_layout_kernel,
        (
            arg0, arg1, arg2, arg3, arg4, arg5,
            arg6, arg7, arg8, arg9, arg10,
            src_flat, clone, partials,
            BLOCK_ROWS, BLOCK_COLS,
        ),
    )
    ct.launch(
        stream,
        (CHANNELS // FINAL_BLOCK_COLS, 1, 1),
        _finalize_kernel,
        (partials, out, num_row_blocks, FINAL_BLOCK_COLS),
    )

    return clone, clone.permute(1, 0), out
