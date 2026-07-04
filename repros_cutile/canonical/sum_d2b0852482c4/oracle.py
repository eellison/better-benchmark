"""cuTile port of sum_d2b0852482c4: Albert 3-add + column reduction.

Result = x0 + x1 + x2 + x3 (each bf16 add rounds to bf16), materialized as
bf16[M,N]. Column sum reduces down the row axis then bf16-rounds and returns
as f32[N]. Returns (out, out.permute(1,0), sum_out).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add3_materialize_kernel(
    x0_ptr,   # bf16 [M, N]
    x1_ptr,
    x2_ptr,
    x3_ptr,
    out_ptr,  # bf16 [M, N]
    ROW_BLOCK: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    row = ct.bid(0)
    col = ct.bid(1)
    x1 = ct.load(x1_ptr, index=(row, col), shape=(ROW_BLOCK, BLOCK_COLS))
    x0 = ct.load(x0_ptr, index=(row, col), shape=(ROW_BLOCK, BLOCK_COLS))
    v = ct.astype(ct.astype(x1, ct.float32) + ct.astype(x0, ct.float32), ct.bfloat16)
    x2 = ct.load(x2_ptr, index=(row, col), shape=(ROW_BLOCK, BLOCK_COLS))
    v = ct.astype(ct.astype(v, ct.float32) + ct.astype(x2, ct.float32), ct.bfloat16)
    x3 = ct.load(x3_ptr, index=(row, col), shape=(ROW_BLOCK, BLOCK_COLS))
    v = ct.astype(ct.astype(v, ct.float32) + ct.astype(x3, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(row, col), tile=v)


@ct.kernel
def _column_sum_kernel(
    out_ptr,      # bf16 [M, N]
    sum_ptr,      # f32 [N]
    M_: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(out_ptr, index=(0, col_block), shape=(M_, BLOCK_COLS))
    values_f = ct.astype(values, ct.float32)
    totals = ct.sum(values_f, axis=0)
    rounded = ct.astype(ct.astype(totals, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=rounded)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="d247a9e9", ROW_BLOCK=128, BLOCK_COLS=64)
@oracle_impl(hardware="B200", point="1b39f873", ROW_BLOCK=256, BLOCK_COLS=32)
def oracle_forward(inputs, *, ROW_BLOCK, BLOCK_COLS):
    x0, x1, x2, x3, _shape0, _shape1, _shape2, out_shape, sum_shape = inputs
    m, n = _shape_tuple(out_shape)

    out = torch.empty_strided((m, n), (n, 1), device=x0.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        _shape_tuple(sum_shape),
        (1,),
        device=x0.device,
        dtype=torch.float32,
    )

    # x1 is (8, 512, 4096) or similar - reshape to (m, n)
    x1_2d = x1.reshape(m, n)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(m, ROW_BLOCK), ct.cdiv(n, BLOCK_COLS), 1),
        _add3_materialize_kernel,
        (x0, x1_2d, x2, x3, out, ROW_BLOCK, BLOCK_COLS),
    )
    # Choose BLOCK_COLS that yields a full column tile in shape (m, BLOCK_COLS)
    # For the reduction, we can use the same BLOCK_COLS or use N which may
    # be too big — but m is up to 32768. We should use a reasonable BLOCK_COLS.
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_COLS), 1, 1),
        _column_sum_kernel,
        (out, sum_out, m, BLOCK_COLS),
    )
    return out, torch.as_strided(out, (n, m), (1, n)), sum_out
