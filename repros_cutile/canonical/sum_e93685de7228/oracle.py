"""cuTile port of sum_e93685de7228: GPT-2 QKV cat + column sum.

Approach:
1. Do the permute+view+cat using torch (layout-only).
2. Run a cuTile column-sum kernel over the resulting [rows, 2304] bf16 tensor.
3. Cast column sum to bf16 and back to f32 to reproduce eager's rounding boundary.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _column_sum_kernel(
    x_arr,      # bf16 [ROWS, COLS]
    out_arr,    # f32 [COLS]
    NUM_ROW_TILES: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    col_tile = ct.bid(0)
    acc = ct.zeros(shape=(COL_BLOCK,), dtype=ct.float32)
    for row_tile in ct.static_iter(range(NUM_ROW_TILES)):
        x = ct.load(x_arr, index=(row_tile, col_tile), shape=(ROW_BLOCK, COL_BLOCK))
        x_f = ct.astype(x, ct.float32)
        row_sum = ct.sum(x_f, axis=0)
        acc = acc + row_sum
    acc_bf = ct.astype(acc, ct.bfloat16)
    acc_f = ct.astype(acc_bf, ct.float32)
    ct.store(out_arr, index=(col_tile,), tile=acc_f)


def _run(inputs, *, ROW_BLOCK, COL_BLOCK, FINAL_COL_BLOCK):
    arg0_1, arg1_1, arg2_1, *_shapes = inputs
    # permute [B, H, S, D] -> [B, S, H, D], then view as [B, S, H*D]
    B, H, S, D = arg0_1.shape
    q = arg0_1.permute(0, 2, 1, 3).contiguous().view(B, S, H * D)
    k = arg1_1.permute(0, 2, 1, 3).contiguous().view(B, S, H * D)
    v = arg2_1.permute(0, 2, 1, 3).contiguous().view(B, S, H * D)
    cat = torch.cat([q, v, k], dim=2)  # [B, S, 3*H*D]
    rows = B * S
    cols = 3 * H * D
    view3 = cat.view(rows, cols)

    stream = torch.cuda.current_stream()
    sum_out = torch.empty((cols,), device=arg0_1.device, dtype=torch.float32)
    num_row_tiles = (rows + ROW_BLOCK - 1) // ROW_BLOCK
    ct.launch(
        stream, (ct.cdiv(cols, COL_BLOCK), 1, 1),
        _column_sum_kernel,
        (view3, sum_out, num_row_tiles, COL_BLOCK, ROW_BLOCK),
    )
    return view3, sum_out


@oracle_impl(hardware="B200", point="1a3ec418", ROW_BLOCK=64, COL_BLOCK=32, FINAL_COL_BLOCK=32)
@oracle_impl(hardware="B200", point="ce8e60c0", ROW_BLOCK=64, COL_BLOCK=32, FINAL_COL_BLOCK=32)
def oracle_forward(inputs, *, ROW_BLOCK, COL_BLOCK, FINAL_COL_BLOCK):
    return _run(inputs, ROW_BLOCK=ROW_BLOCK, COL_BLOCK=COL_BLOCK, FINAL_COL_BLOCK=FINAL_COL_BLOCK)
