"""cuTile port of pointwise_f22251f83eba: XLNet flat transpose (16384, D) -> (D, 16384)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUT_COLS = 1024
GROUPS = 16


@ct.kernel
def _xlnet_flat_transpose_kernel(
    src,  # bf16 (16384, D)
    dst,  # bf16 (D, 16384)
    YBLOCK: ct.Constant[int],
    XBLOCK: ct.Constant[int],
):
    y_block = ct.bid(0)
    x_block = ct.bid(1)
    # Load a (XBLOCK, YBLOCK) tile from source at (x_block, y_block).
    vals = ct.load(src, index=(x_block, y_block), shape=(XBLOCK, YBLOCK))
    # Transpose to (YBLOCK, XBLOCK) for the destination.
    vals_t = ct.transpose(vals)
    ct.store(dst, index=(y_block, x_block), tile=vals_t)


@oracle_impl(hardware="B200", point="144bae60", YBLOCK=16, XBLOCK=128)
@oracle_impl(hardware="B200", point="e95c7520", YBLOCK=16, XBLOCK=128)
def oracle_forward(inputs, *, YBLOCK, XBLOCK):
    x = inputs[0]
    d = int(x.shape[2])
    out = torch.empty_strided((d * GROUPS, OUT_COLS), (OUT_COLS, 1), device=x.device, dtype=x.dtype)
    src_2d = x.view(16384, d)
    dst_2d = out.view(d, 16384)
    stream = torch.cuda.current_stream()
    grid = ((d + YBLOCK - 1) // YBLOCK, (16384 + XBLOCK - 1) // XBLOCK, 1)
    ct.launch(
        stream,
        grid,
        _xlnet_flat_transpose_kernel,
        (src_2d, dst_2d, YBLOCK, XBLOCK),
    )
    return out
