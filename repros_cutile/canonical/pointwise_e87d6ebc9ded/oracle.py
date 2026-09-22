"""cuTile port of pointwise_e87d6ebc9ded: attention head-major to
sequence-major layout materialization.

The op is `view(B,H,S,D).permute(0,2,1,3).contiguous().view(B*S,H*D)`.

Match the Triton oracle's structure: a single 2D-tile kernel that copies a
(BLOCK_S, HEAD_DIM) block per program (grid = (batch, heads, seq/BLOCK_S)),
mirroring Triton's BLOCK_ROWS x BLOCK_COLS output-space tile. For non-power-
of-2 HEAD_DIM (e.g. 80), fall back to a scatter-based kernel that masks the
tail with padding_mode=ZERO on load.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _head_layout_copy_kernel(
    src,   # (B, H, S, D)
    dst,   # (rows, hidden) where hidden = H*D, rows=B*S
    SEQ: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    batch = ct.bid(0)
    head = ct.bid(1)
    seq_block = ct.bid(2)
    # Load BLOCK_S consecutive seq entries for this (batch, head).
    x = ct.load(src, index=(batch, head, seq_block, 0),
                shape=(1, 1, BLOCK_S, HEAD_DIM))
    # Reshape (1,1,BLOCK_S,HEAD_DIM) -> (BLOCK_S, HEAD_DIM).
    y = ct.reshape(x, (BLOCK_S, HEAD_DIM))
    # Output tile-space: row index = batch*(SEQ/BLOCK_S) + seq_block,
    # col index = head. In tile-space coords this is:
    row_tile = batch * (SEQ // BLOCK_S) + seq_block
    ct.store(dst, index=(row_tile, head), tile=y)


@ct.kernel
def _head_layout_copy_scatter_kernel(
    src,   # (B, H, S, D)
    dst,   # (rows, hidden) where hidden = H*D, rows=B*S
    SEQ: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    HEAD_DIM_PAD: ct.Constant[int],
):
    batch = ct.bid(0)
    head = ct.bid(1)
    seq = ct.bid(2)
    x = ct.load(
        src,
        index=(batch, head, seq, 0),
        shape=(1, 1, 1, HEAD_DIM_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row = batch * SEQ + seq
    y = ct.reshape(x, (HEAD_DIM_PAD,))
    idx = ct.arange(HEAD_DIM_PAD, dtype=ct.int32)
    col_idx = head * HEAD_DIM + idx
    row_idx = ct.full((HEAD_DIM_PAD,), fill_value=row, dtype=ct.int32)
    mask = idx < HEAD_DIM
    ct.scatter(dst, (row_idx, col_idx), y, mask=mask)


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@oracle_impl(hardware="B200", point="fb089404")
@oracle_impl(hardware="B200", point="226fbbfa")
@oracle_impl(hardware="B200", point="c6b0f684")
@oracle_impl(hardware="B200", point="07e248d7")
@oracle_impl(hardware="B200", point="576ca76e")
@oracle_impl(hardware="B200", point="471d82af")
@oracle_impl(hardware="B200", point="c6cb1dd8")
@oracle_impl(hardware="B200", point="e6f344ac")
@oracle_impl(hardware="B200", point="2cdbce9d")
@oracle_impl(hardware="B200", point="c23ba4e7")
@oracle_impl(hardware="B200", point="6c3c2efc")
@oracle_impl(hardware="B200", point="a3cab238")
@oracle_impl(hardware="B200", point="14c0be85")
@oracle_impl(hardware="B200", point="d528e08b")
@oracle_impl(hardware="B200", point="1dcf8636")
def oracle_forward(inputs):
    arg0, shape0, _shape1, shape2 = inputs
    heads = int(shape0[1])
    seq = int(arg0.shape[1])
    head_dim = int(arg0.shape[2])
    rows = int(shape2[0])
    batch = rows // seq

    # Reshape input from (B*H, S, D) to (B, H, S, D)
    src_view = arg0.view(batch, heads, seq, head_dim)

    output = torch.empty_strided(
        (rows, int(shape2[1])),
        (int(shape2[1]), 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )

    stream = torch.cuda.current_stream()
    if head_dim & (head_dim - 1) == 0:
        # Choose BLOCK_S so that BLOCK_S * HEAD_DIM is as close to Triton's
        # tile size (~4096 elements) as possible, and BLOCK_S divides SEQ.
        target = 4096
        block_s = max(1, min(seq, target // head_dim))
        block_s = _next_pow2(block_s)
        if block_s > seq:
            block_s = seq
        # Ensure divisibility of seq by block_s (seq is power-of-2 in this corpus).
        while seq % block_s != 0 and block_s > 1:
            block_s //= 2
        ct.launch(
            stream,
            (batch, heads, seq // block_s),
            _head_layout_copy_kernel,
            (src_view, output, seq, head_dim, block_s),
        )
    else:
        # For non-power-of-2 head_dim (e.g. 80), round up to next p2.
        head_dim_pad = 1 << (head_dim - 1).bit_length()
        ct.launch(
            stream,
            (batch, heads, seq),
            _head_layout_copy_scatter_kernel,
            (src_view, output, seq, head_dim, head_dim_pad),
        )
    return output
