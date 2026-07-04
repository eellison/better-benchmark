"""cuTile port of pointwise_c8d23ac4414d: bf16 bottom-row constant_pad_nd."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_2d_kernel(src, dst, BLOCK_N: ct.Constant[int]):
    row = ct.bid(0)
    col_block = ct.bid(1)
    v = ct.load(src, index=(row, col_block), shape=(1, BLOCK_N))
    ct.store(dst, index=(row, col_block), tile=v)


def _launch_bottom_pad(inputs, *, BLOCK_N: int):
    x, pad = inputs
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    pad_rows = int(pad[-1])
    out_rows = rows + pad_rows
    # torch.zeros pre-fills zero rows; cuTile kernel copies the input prefix.
    out = torch.zeros((out_rows, cols), device=x.device, dtype=torch.bfloat16)
    # Use contiguous view of input (strided inputs handled by materializing).
    x_c = x.contiguous()
    dst_view = out[:rows]  # first rows of out; same col stride as out.
    stream = torch.cuda.current_stream()
    grid = (rows, cols // BLOCK_N, 1)
    ct.launch(stream, grid, _copy_2d_kernel, (x_c, dst_view, BLOCK_N))
    return out


# BLOCK_N=128 divides 768, 1024, 1536, 128, 2048, 4096, 512.
@oracle_impl(hardware="B200", point="ba1b8f0f", BLOCK_N=128)
@oracle_impl(hardware="B200", point="6883fad3", BLOCK_N=128)
@oracle_impl(hardware="B200", point="1779a8cb", BLOCK_N=128)
@oracle_impl(hardware="B200", point="cd997de8", BLOCK_N=128)
@oracle_impl(hardware="B200", point="cb779bb6", BLOCK_N=128)
@oracle_impl(hardware="B200", point="10c5c015", BLOCK_N=128)
@oracle_impl(hardware="B200", point="d67c38a5", BLOCK_N=128)
@oracle_impl(hardware="B200", point="360d77c3", BLOCK_N=128)
@oracle_impl(hardware="B200", point="d59edba9", BLOCK_N=128)
@oracle_impl(hardware="B200", point="ea31889c", BLOCK_N=128)
@oracle_impl(hardware="B200", point="4c38b93b", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    return _launch_bottom_pad(inputs, BLOCK_N=BLOCK_N)
