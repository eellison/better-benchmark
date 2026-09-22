"""cuTile port of sum_b822d095bf9f: selected-slice row L2 normalization.

For row in [0, BATCH): read x[row, SELECTED_TIMESTEP, :] (bf16, N_COLS=256),
compute row norm (fp32 sqrt of sum_sq), clamp min 1e-12, and produce normalized row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
TIMESTEPS = 50
COLS = 256
SELECTED_TIMESTEP = 49


@ct.kernel
def _selected_row_norm_kernel(
    x_ptr,      # bf16 [BATCH, TIMESTEPS, COLS]
    norm_ptr,   # f32 [BATCH]
    div_ptr,    # f32 [BATCH, COLS]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    # Load one row: x[row, SELECTED_TIMESTEP, :]
    x = ct.load(x_ptr, index=(row, SELECTED_TIMESTEP, 0), shape=(1, 1, BLOCK_N))
    x_f = ct.astype(x, ct.float32)
    x_2d = ct.reshape(x_f, (1, BLOCK_N))

    sum_sq = ct.sum(x_2d * x_2d)
    norm = ct.sqrt(sum_sq)
    denom = ct.where(norm < 1.0e-12, ct.full(shape=(), fill_value=1.0e-12, dtype=ct.float32), norm)
    out = x_2d * (1.0 / denom)

    norm_tile = ct.full(shape=(1,), fill_value=norm, dtype=ct.float32)
    ct.store(norm_ptr, index=(row,), tile=norm_tile)
    ct.store(div_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="6cbb208b", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, shape_param = inputs
    selected = torch.as_strided(
        x,
        (BATCH, COLS),
        (TIMESTEPS * COLS, 1),
        storage_offset=SELECTED_TIMESTEP * COLS,
    )
    norm = torch.empty_strided((BATCH, 1), (1, 1), device=x.device, dtype=torch.float32)
    div = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    norm_flat = norm.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _selected_row_norm_kernel,
        (x, norm_flat, div, BLOCK_N),
    )
    return selected, norm, div
