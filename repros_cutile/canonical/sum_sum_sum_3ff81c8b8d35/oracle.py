"""cuTile port of sum_sum_sum_3ff81c8b8d35: XLNet LayerNorm backward + dropout.

For each row:
  x = (arg0 + arg2 + arg3) as bf16-add, cast to f32, then + arg1 (f32).
  Actually eager order: add(arg1, cast(arg0)), then add(., cast(arg2)),
  add(., cast(arg3)). So all four contributions are summed.
  weighted = x * weight, row_sum, row_dot = sum(weighted), sum(weighted*rhs)
  centered = weighted*1024 - row_sum - rhs*row_dot
  grad = scale * centered
  drop_bf16 = bf16(grad) * bf16(keep * 1.1111)
Outputs:
  grad_out (f32 [512,16,1024]), out_x_rhs (f32 [1024]), out_x (f32 [1024]),
  drop_out (bf16 [8192,1024]), drop_out.T, out_drop (f32 [1024]).

HIDDEN=1024 fits one row-tile of BLOCK_C=1024. ROWS=8192.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 1024
DROP_SCALE = 1.1111111111111112


@ct.kernel
def _row_kernel(
    arg0_ptr,     # bf16 [ROWS, HIDDEN]
    arg1_ptr,     # f32  [ROWS, HIDDEN]
    arg2_ptr,     # bf16 [ROWS, HIDDEN]
    arg3_ptr,     # bf16 [ROWS, HIDDEN]
    weight_ptr,   # f32 [HIDDEN]
    rhs_ptr,      # f32 [ROWS, HIDDEN]
    scale_ptr,    # f32 [ROWS]
    keep_ptr,     # b8  [ROWS, HIDDEN]
    grad_out_ptr, # f32 [ROWS, HIDDEN]
    drop_out_ptr, # bf16 [ROWS, HIDDEN]
    partial_x_rhs_ptr, # f32 [ROWS, HIDDEN]
    partial_x_ptr,     # f32 [ROWS, HIDDEN]
    partial_drop_ptr,  # f32 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
):
    row = ct.bid(0)
    a0 = ct.load(arg0_ptr, index=(row, 0), shape=(1, HIDDEN_))
    a1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, HIDDEN_))
    a2 = ct.load(arg2_ptr, index=(row, 0), shape=(1, HIDDEN_))
    a3 = ct.load(arg3_ptr, index=(row, 0), shape=(1, HIDDEN_))

    x = a1 + ct.astype(a0, ct.float32)
    x = x + ct.astype(a2, ct.float32)
    x = x + ct.astype(a3, ct.float32)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))
    weighted = x * weight_2d
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, HIDDEN_))

    row_sum = ct.sum(weighted)
    row_dot = ct.sum(weighted * rhs)
    centered = weighted * float(HIDDEN_) - row_sum - rhs * row_dot

    scale = ct.load(scale_ptr, index=(row,), shape=(1,))
    scale_2d = ct.reshape(scale, (1, 1))
    grad = scale_2d * centered
    ct.store(grad_out_ptr, index=(row, 0), tile=grad)

    grad_bf16 = ct.astype(grad, ct.bfloat16)
    keep = ct.load(keep_ptr, index=(row, 0), shape=(1, HIDDEN_))
    keep_f = ct.astype(keep, ct.float32)
    keep_scale_bf16 = ct.astype(keep_f * DROP_SCALE, ct.bfloat16)
    # Two candidates in Triton kernel: `drop_bf16` used for reduction only, and
    # `store_drop_bf16` used for the stored bf16 tensor. Reduction path uses
    # bf16(grad) * bf16(keep_scale) then bf16; store path uses f32(grad) *
    # bf16(keep_scale) then bf16.
    drop_reduction_bf16 = ct.astype(
        ct.astype(grad_bf16, ct.float32) * ct.astype(keep_scale_bf16, ct.float32),
        ct.bfloat16,
    )
    store_drop_bf16 = ct.astype(grad * ct.astype(keep_scale_bf16, ct.float32), ct.bfloat16)
    ct.store(drop_out_ptr, index=(row, 0), tile=store_drop_bf16)

    # Per-row column contributions to global reductions. We write them out into
    # a scratch [ROWS, HIDDEN] and do a two-stage sum externally.
    ct.store(partial_x_rhs_ptr, index=(row, 0), tile=x * rhs)
    ct.store(partial_x_ptr, index=(row, 0), tile=x)
    ct.store(partial_drop_ptr, index=(row, 0), tile=ct.astype(drop_reduction_bf16, ct.float32))


@ct.kernel
def _column_partial_kernel(
    src_ptr,          # f32 [ROWS, HIDDEN]
    partial_ptr,      # f32 [num_row_tiles, HIDDEN]
    ROW_BLOCK: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    values = ct.load(src_ptr, index=(row_tile, col_tile), shape=(ROW_BLOCK, COL_BLOCK))
    partial = ct.sum(values, axis=0)
    ct.store(partial_ptr, index=(row_tile, col_tile), tile=ct.reshape(partial, (1, COL_BLOCK)))


@ct.kernel
def _column_finalize_kernel(
    partial_ptr,      # f32 [num_row_tiles, HIDDEN]
    out_ptr,          # f32 [HIDDEN]
    NUM_ROW_TILES: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
    ROW_TILE_BLOCK: ct.Constant[int],
):
    col_tile = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(0, col_tile), shape=(ROW_TILE_BLOCK, COL_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_idx = ct.arange(ROW_TILE_BLOCK, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_ROW_TILES, (ROW_TILE_BLOCK, 1))
    valid = ct.where(row_mask, values, 0.0)
    total = ct.sum(valid, axis=0)
    ct.store(out_ptr, index=(col_tile,), tile=total)


@ct.kernel
def _column_finalize_bf16_kernel(
    partial_ptr,      # f32 [num_row_tiles, HIDDEN]
    out_ptr,          # f32 [HIDDEN]
    NUM_ROW_TILES: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
    ROW_TILE_BLOCK: ct.Constant[int],
):
    col_tile = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(0, col_tile), shape=(ROW_TILE_BLOCK, COL_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_idx = ct.arange(ROW_TILE_BLOCK, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_ROW_TILES, (ROW_TILE_BLOCK, 1))
    valid = ct.where(row_mask, values, 0.0)
    total = ct.sum(valid, axis=0)
    total_bf16 = ct.astype(total, ct.bfloat16)
    total_f32 = ct.astype(total_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=total_f32)


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


ROW_BLOCK_COL = 128
COL_BLOCK = 128


@oracle_impl(
    hardware="B200",
    point="6d721589",
    ROWS_PER_GROUP=16,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=8,
)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    assert hidden == HIDDEN
    device = arg0_1.device

    full_shape = tuple(int(d) for d in arg1_1.shape)  # (512, 16, 1024)
    flat_shape = tuple(int(d) for d in _shape_param_3)  # (8192, 1024)

    # 3D inputs viewed as [rows, hidden].
    arg1_flat = arg1_1.view(rows, hidden)
    arg5_flat = arg5_1.view(rows, hidden)
    arg6_flat = arg6_1.view(rows)
    arg7_flat = arg7_1.view(rows, hidden)

    grad_flat = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    drop_out = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device,
        dtype=torch.bfloat16,
    )
    part_x_rhs = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    part_x = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    part_drop = torch.empty((rows, hidden), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _row_kernel,
        (arg0_1, arg1_flat, arg2_1, arg3_1, arg4_1, arg5_flat, arg6_flat,
         arg7_flat, grad_flat, drop_out, part_x_rhs, part_x, part_drop, hidden),
    )

    # Column reduce.
    num_row_tiles = rows // ROW_BLOCK_COL
    assert rows % ROW_BLOCK_COL == 0
    num_col_tiles = hidden // COL_BLOCK
    assert hidden % COL_BLOCK == 0

    tmp_x_rhs = torch.empty((num_row_tiles, hidden), device=device, dtype=torch.float32)
    tmp_x = torch.empty((num_row_tiles, hidden), device=device, dtype=torch.float32)
    tmp_drop = torch.empty((num_row_tiles, hidden), device=device, dtype=torch.float32)

    ct.launch(
        stream, (num_row_tiles, num_col_tiles, 1), _column_partial_kernel,
        (part_x_rhs, tmp_x_rhs, ROW_BLOCK_COL, COL_BLOCK),
    )
    ct.launch(
        stream, (num_row_tiles, num_col_tiles, 1), _column_partial_kernel,
        (part_x, tmp_x, ROW_BLOCK_COL, COL_BLOCK),
    )
    ct.launch(
        stream, (num_row_tiles, num_col_tiles, 1), _column_partial_kernel,
        (part_drop, tmp_drop, ROW_BLOCK_COL, COL_BLOCK),
    )

    out_x_rhs = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_drop = torch.empty((hidden,), device=device, dtype=torch.float32)

    ROW_TILE_BLOCK = _next_p2(num_row_tiles)
    ct.launch(
        stream, (num_col_tiles, 1, 1), _column_finalize_kernel,
        (tmp_x_rhs, out_x_rhs, num_row_tiles, COL_BLOCK, ROW_TILE_BLOCK),
    )
    ct.launch(
        stream, (num_col_tiles, 1, 1), _column_finalize_kernel,
        (tmp_x, out_x, num_row_tiles, COL_BLOCK, ROW_TILE_BLOCK),
    )
    ct.launch(
        stream, (num_col_tiles, 1, 1), _column_finalize_bf16_kernel,
        (tmp_drop, out_drop, num_row_tiles, COL_BLOCK, ROW_TILE_BLOCK),
    )

    # Match the fp32 out_x_rhs/out_x layouts. sum_shape is (1024,) in this
    # capture.
    grad_out_full = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device,
        dtype=torch.float32,
    )
    grad_out_full.view(rows, hidden).copy_(grad_flat)

    # Compose sibling `out_x_rhs`, `out_x` with the (1,) stride the eager
    # reference returns.
    x_rhs_out = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    x_out = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    drop_scalar_out = torch.empty_strided((hidden,), (1,), device=device, dtype=torch.float32)
    x_rhs_out.copy_(out_x_rhs)
    x_out.copy_(out_x)
    drop_scalar_out.copy_(out_drop)

    return grad_out_full, x_rhs_out, x_out, drop_out, drop_out.permute(1, 0), drop_scalar_out
