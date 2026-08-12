"""cuTile port of pointwise_6fa7595f145a: Longformer iota+as_strided+clone as an index-map materialization."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DIM0 = 96
WINDOWS = 4
WINDOW = 768
HEAD_DIM = 64
OUTER = DIM0 * WINDOWS
NUMEL = OUTER * WINDOW * HEAD_DIM
OUTER_STRIDE = WINDOW * HEAD_DIM
SOURCE_STRIDE_DIM0 = 98304
SOURCE_STRIDE_WINDOW = 16384
SOURCE_STRIDE_POS = 64


@ct.kernel
def _longformer_iota4_clone_kernel(
    out_ptr,  # (OUTER, WINDOW, HEAD_DIM) i64
    BLOCK_W: ct.Constant[int],
    HEAD_DIM_: ct.Constant[int],
    SS0: ct.Constant[int],
    SS1: ct.Constant[int],
    SS2: ct.Constant[int],
):
    outer = ct.bid(0)
    window_block = ct.bid(1)

    # For a fixed outer=(dim0, window_id), stores (pos, dim) tile.
    dim0 = outer // 4
    window_id = outer - dim0 * 4
    pos = window_block * BLOCK_W + ct.arange(BLOCK_W, dtype=ct.int64)
    dim = ct.arange(HEAD_DIM_, dtype=ct.int64)

    pos_2d = ct.reshape(pos, (BLOCK_W, 1))
    dim_2d = ct.reshape(dim, (1, HEAD_DIM_))

    values = (
        dim0 * SS0
        + window_id * SS1
        + pos_2d * SS2
        + dim_2d
    )
    ct.store(out_ptr, index=(outer, window_block, 0), tile=ct.reshape(values, (1, BLOCK_W, HEAD_DIM_)))


@oracle_impl(hardware="B200", point="d7517139", BLOCK_W=64)
def oracle_forward(inputs, *, BLOCK_W):
    del inputs
    out = torch.empty_strided(
        (NUMEL,),
        (1,),
        device=torch.device("cuda", 0),
        dtype=torch.int64,
    )
    out_3d = out.view(OUTER, WINDOW, HEAD_DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (OUTER, WINDOW // BLOCK_W, 1),
        _longformer_iota4_clone_kernel,
        (out_3d, BLOCK_W, HEAD_DIM, SOURCE_STRIDE_DIM0, SOURCE_STRIDE_WINDOW, SOURCE_STRIDE_POS),
    )
    return out
