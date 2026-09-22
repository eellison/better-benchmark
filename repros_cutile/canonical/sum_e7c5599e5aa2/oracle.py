"""cuTile port of sum_e7c5599e5aa2: MobileViT QKV pad/cat/clone + column sum.

SCHEDULER_FUSION: pad three bf16 tensors along the last dim (with possibly
negative padding = slice), cat them along dim 0, view/permute/clone to a
contiguous [rows, cols] output. Return that matrix, its transpose alias, and
its fp32 column sum with a bf16 roundtrip.

We do the pad/cat/permute in torch (matching the eager Repro) and use cuTile
for the final materialization copy and column reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_kernel(
    src_ptr,   # bf16 [rows, cols]
    dst_ptr,   # bf16 [rows, cols]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    values = ct.load(src_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N))
    ct.store(dst_ptr, index=(row_tile, col_tile), tile=values)


@ct.kernel
def _col_sum_bf16_kernel(
    x_ptr,      # bf16 [rows, cols]
    out_ptr,    # f32 [cols]
    ROWS_BLOCK: ct.Constant[int],
    NUM_ROW_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    zero = ct.zeros((BLOCK_N,), dtype=ct.float32)
    acc = zero
    for i in range(NUM_ROW_TILES):
        values = ct.load(x_ptr, index=(i, col_tile), shape=(ROWS_BLOCK, BLOCK_N))
        values_f = ct.astype(values, ct.float32)
        acc = acc + ct.sum(values_f, axis=0)
    rounded_bf16 = ct.astype(acc, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=rounded_f32)


@oracle_impl(
    hardware="B200",
    point="89a27d2d",
    ROWS_BLOCK=512,
    BLOCK_N=16,
)
@oracle_impl(
    hardware="B200",
    point="38bf62e8",
    ROWS_BLOCK=128,
    BLOCK_N=16,
)
def oracle_forward(inputs, *, ROWS_BLOCK: int, BLOCK_N: int):
    x0, x1, x2, pad0, _pad1, _pad2, view_shape, _view_shape_1, view_shape_2, _sum_shape = inputs
    # Pad amounts from shape param. pad0 is [left_pad, right_pad] for last dim.
    pad0_list = [int(v) for v in pad0]

    def pad(t):
        return torch.constant_pad_nd(t, pad0_list, 0.0)

    x0p = pad(x0)
    x1p = pad(x1)
    x2p = pad(x2)
    catted = torch.cat([x2p, x1p, x0p], dim=0)
    viewed = catted.view(*[int(v) for v in view_shape])
    permuted = viewed.permute(1, 3, 0, 2, 4).contiguous()
    matrix_shape_2d = [int(v) for v in view_shape_2]
    matrix_src = permuted.view(*matrix_shape_2d)
    rows, cols = matrix_src.shape

    matrix = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((cols,), device=x0.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    BLOCK_M = ROWS_BLOCK
    num_row_tiles = rows // BLOCK_M
    ct.launch(
        stream,
        (num_row_tiles, cols // BLOCK_N, 1),
        _copy_kernel,
        (matrix_src, matrix, BLOCK_M, BLOCK_N),
    )
    ct.launch(
        stream,
        (cols // BLOCK_N, 1, 1),
        _col_sum_bf16_kernel,
        (matrix, sum_out, BLOCK_M, num_row_tiles, BLOCK_N),
    )
    return matrix, matrix.permute(1, 0), sum_out
