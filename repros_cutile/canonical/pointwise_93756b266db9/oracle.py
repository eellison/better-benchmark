"""cuTile port of pointwise_93756b266db9: SigLIP bias add [128,768] + bias[768]."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROW_SIZE = 768
ROWS = 128


@ct.kernel
def _bias_add_kernel(x_ptr, bias_ptr, out_ptr, BLOCK: ct.Constant[int], ROW: ct.Constant[int]):
    # 1D tile of size BLOCK; row-major index
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    # cols per element: offsets % ROW. Since BLOCK <= ROW (256 < 768) and ROW % BLOCK == 0,
    # each tile spans a contiguous slice within ONE row (or wraps a row boundary at BLOCK==256 * k
    # ... 768/256=3, so each tile lies entirely within one row). Compute bias offset for tile start.
    # Actually easier: reshape flat index space to (ROWS, ROW) and use bid across rows and cols.
    # Handled below.
    pass


@ct.kernel
def _bias_add_2d_kernel(
    x_ptr,      # bf16 [ROWS, ROW_SIZE]
    bias_ptr,   # bf16 [ROW_SIZE]
    out_ptr,    # bf16 [ROWS, ROW_SIZE]
    ROW_BLOCK: ct.Constant[int],
    COL_BLOCK: ct.Constant[int],
):
    r = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(x_ptr, index=(r, c), shape=(ROW_BLOCK, COL_BLOCK))
    b = ct.load(bias_ptr, index=(c,), shape=(COL_BLOCK,))
    b2 = ct.reshape(b, (1, COL_BLOCK))
    xf = ct.astype(x, ct.float32)
    bf = ct.astype(b2, ct.float32)
    y = ct.astype(xf + bf, ct.bfloat16)
    ct.store(out_ptr, index=(r, c), tile=y)


@oracle_impl(hardware="B200", point="a6d7e457", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, *_shape_params = inputs
    backing = torch.empty_strided(
        (128, 1, 12, 64),
        (768, 768, 64, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # Reshape backing to [ROWS, ROW_SIZE]
    backing_2d = torch.as_strided(backing, (ROWS, ROW_SIZE), (ROW_SIZE, 1))
    x_2d = arg0_1  # already [128, 768]
    stream = torch.cuda.current_stream()
    ROW_BLOCK = 1
    COL_BLOCK = BLOCK  # BLOCK divides ROW_SIZE (768 / 256 = 3)
    ct.launch(
        stream,
        (ROWS // ROW_BLOCK, ROW_SIZE // COL_BLOCK, 1),
        _bias_add_2d_kernel,
        (x_2d, arg1_1, backing_2d, ROW_BLOCK, COL_BLOCK),
    )
    return backing.permute(0, 2, 1, 3)
