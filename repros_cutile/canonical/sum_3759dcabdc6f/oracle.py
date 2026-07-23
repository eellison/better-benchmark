"""cuTile port of sum_3759dcabdc6f: layout clone + column sum with bf16 roundtrip.

Uses torch to build the [16384, 1024]-style contiguous layout (permute+clone),
then a cuTile kernel to do the column sum plus bf16 rounding.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _col_sum_bf16_roundtrip_kernel(
    input_ptr,     # bf16 [B, C]
    out_ptr,       # f32 [C]
    B: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_tile = ct.bid(0)
    # Load a (BLOCK_B, BLOCK_C) tile, reduce over B
    tile = ct.load(input_ptr, index=(0, col_tile), shape=(BLOCK_B, BLOCK_C))
    tile_f = ct.astype(tile, ct.float32)
    col_sum = ct.sum(tile_f, axis=0)  # (BLOCK_C,)
    rounded_bf16 = ct.astype(col_sum, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=rounded_f32)


@oracle_impl(hardware="B200", point="2c5b25cf", BLOCK_C=64)
@oracle_impl(hardware="B200", point="687c0b28", BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_C):
    x, view_shape, out2d_shape, flat_shape, sum_shape = inputs

    # Compute the layout: permute+view+permute+view+clone equivalent to
    # x.permute(0, 2, 1).view(view_shape).permute(0, 2, 1, 3).view(out2d_shape).clone().view(flat_shape)
    view_shape = tuple(int(d) for d in view_shape)
    out2d_shape = tuple(int(d) for d in out2d_shape)
    flat_shape = tuple(int(d) for d in flat_shape)

    layout = (
        x.permute(0, 2, 1)
        .contiguous()
        .view(view_shape)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(out2d_shape)
        .view(flat_shape)
    )
    # layout: bf16 [B, C]
    B = int(layout.shape[0])
    C = int(layout.shape[1])

    out_flat = torch.empty(C, device=x.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C // BLOCK_C, 1, 1),
        _col_sum_bf16_roundtrip_kernel,
        (layout, out_flat, B, C, B, BLOCK_C),
    )
    return (layout, layout.permute(1, 0), out_flat)
