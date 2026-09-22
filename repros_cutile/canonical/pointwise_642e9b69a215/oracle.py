"""cuTile port of pointwise_642e9b69a215: ConvBert head-concat of two 384-column halves."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_concat_left_kernel(src_ptr, dst_ptr, BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int]):
    row = ct.bid(1)
    col = ct.bid(0)
    tile = ct.load(src_ptr, index=(row, col), shape=(BLOCK_M, BLOCK_N))
    ct.store(dst_ptr, index=(row, col), tile=tile)


@oracle_impl(hardware="B200", point="add2068b", BLOCK_M=2, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    arg0_1, arg1_1, _s0, _s1, _s2, _shape3 = inputs
    n = int(_shape3[0])
    width = int(_shape3[1])
    # arg0_1 has strides that lay it out as [16384, 384] contiguous in storage
    # (already permuted logical). Similarly arg1_1 is [16384, 384] worth of contiguous data.
    src0 = torch.as_strided(arg0_1, (n, 384), (384, 1))
    src1 = torch.as_strided(arg1_1, (n, 384), (384, 1))
    out = torch.empty_strided((n, width), (width, 1), device=arg0_1.device, dtype=arg0_1.dtype)
    left = out[:, :384]
    right = out[:, 384:]
    stream = torch.cuda.current_stream()
    grid_rows = ct.cdiv(n, BLOCK_M)
    grid_cols = ct.cdiv(384, BLOCK_N)
    ct.launch(stream, (grid_cols, grid_rows, 1), _head_concat_left_kernel,
              (src0, left, BLOCK_M, BLOCK_N))
    ct.launch(stream, (grid_cols, grid_rows, 1), _head_concat_left_kernel,
              (src1, right, BLOCK_M, BLOCK_N))
    return out
