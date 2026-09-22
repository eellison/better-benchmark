"""cuTile port of sum_e70c30104d29: M2M100 masked bf16 view + column sum + f32 cast.

Two kernels: one writes the masked bf16 output tensor (view of arg0),
one accumulates the column sums (dim=0 reduction of 8192 rows over 4096 cols).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 4096
BLOCK_R = 64
BLOCK_C = 128


@ct.kernel
def _masked_where_kernel(
    src_ptr,   # bf16 [ROWS, COLS]
    mask_ptr,  # bool [ROWS, COLS]
    out_ptr,   # bf16 [ROWS, COLS]
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    r_tile = ct.bid(0)
    c_tile = ct.bid(1)

    src = ct.load(src_ptr, index=(r_tile, c_tile), shape=(BLOCK_R_, BLOCK_C_))
    mask = ct.load(mask_ptr, index=(r_tile, c_tile), shape=(BLOCK_R_, BLOCK_C_))
    zero = ct.full(shape=(BLOCK_R_, BLOCK_C_), fill_value=0.0, dtype=ct.bfloat16)
    out = ct.where(mask, zero, src)
    ct.store(out_ptr, index=(r_tile, c_tile), tile=out)


@ct.kernel
def _column_sum_kernel(
    where_ptr,  # bf16 [ROWS, COLS]
    sum_ptr,    # f32 [COLS]
    NUM_R_TILES: ct.Constant[int],
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_tile = ct.bid(0)
    acc = ct.full(shape=(BLOCK_C_,), fill_value=0.0, dtype=ct.float32)
    for r_tile in range(NUM_R_TILES):
        vals = ct.load(where_ptr, index=(r_tile, c_tile), shape=(BLOCK_R_, BLOCK_C_))
        vals_f = ct.astype(vals, ct.float32)
        acc = acc + ct.sum(vals_f, axis=0)
    acc_bf16 = ct.astype(acc, ct.bfloat16)
    acc_f = ct.astype(acc_bf16, ct.float32)
    ct.store(sum_ptr, index=(c_tile,), tile=acc_f)


@oracle_impl(hardware="B200", point="4a8b763e")
def oracle_forward(inputs):
    arg0_1, arg1_1, _s0, _s1, _s2 = inputs
    where_out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((COLS,), (1,), device=arg0_1.device, dtype=torch.float32)

    mask_2d = arg1_1.view(ROWS, COLS)

    stream = torch.cuda.current_stream()
    num_r_tiles = ROWS // BLOCK_R
    num_c_tiles = COLS // BLOCK_C
    ct.launch(
        stream,
        (num_r_tiles, num_c_tiles, 1),
        _masked_where_kernel,
        (arg0_1, mask_2d, where_out, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_tiles, 1, 1),
        _column_sum_kernel,
        (where_out, sum_out, num_r_tiles, BLOCK_R, BLOCK_C),
    )
    permute = where_out.permute(1, 0)
    return where_out, permute, sum_out
