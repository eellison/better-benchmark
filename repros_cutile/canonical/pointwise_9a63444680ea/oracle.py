"""cuTile port of pointwise_9a63444680ea: ConvBert local-window sliding view.

Mirrors the two-kernel Triton structure:
  kernel 1: metadata emission (iota9 + iota512 index base).
  kernel 2: fill the (98304, 64, 9) bf16 window buffer directly from arg0_1
            (32*512, 384). Each program handles one output row, static_iter
            over 9 window slots and loads a (1, 64) tile per slot.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 98304
GROUPS = 6           # 384 / 64
CHANNELS = 64
KWIN = 9
BLOCK_S = 512        # kernel metadata block for indices dim 1
KWIN_PAD = 16        # power-of-2 padding for metadata indices dim 0
BATCH = 32
SEQ = 512


@ct.kernel
def _metadata_kernel(
    full_ptr,          # i64 [1, 1]
    indices_pad_ptr,   # i64 [KWIN_PAD, BLOCK_S] (padded)
):
    ks = ct.arange(KWIN_PAD, dtype=ct.int32)
    ss = ct.arange(BLOCK_S, dtype=ct.int32)
    ks_2d = ct.reshape(ks, (KWIN_PAD, 1))
    ss_2d = ct.reshape(ss, (1, BLOCK_S))
    vals = ct.astype(ks_2d + ss_2d, ct.int64)
    ct.store(indices_pad_ptr, index=(0, 0), tile=vals)
    full_zero = ct.zeros((1, 1), dtype=ct.int64)
    ct.store(full_ptr, index=(0, 0), tile=full_zero)


@ct.kernel
def _convbert_window_kernel(
    x_ptr,     # bf16 [16384, 384]
    out_ptr,   # bf16 [98304, 64, 9]
):
    row = ct.bid(0)  # 0..98303
    group = row % 6
    tmp = row // 6
    seq = tmp % SEQ
    batch = tmp // SEQ
    zero_tile = ct.zeros((1, CHANNELS), dtype=ct.bfloat16)
    for k in ct.static_iter(range(KWIN)):
        rel = seq + k - 4
        valid = (rel >= 0) & (rel < SEQ)
        safe_rel = ct.maximum(rel, 0)
        safe_rel = ct.minimum(safe_rel, SEQ - 1)
        source_row = batch * SEQ + safe_rel
        tile2d = ct.load(
            x_ptr, index=(source_row, group), shape=(1, CHANNELS)
        )
        tile2d = ct.where(valid, tile2d, zero_tile)
        ct.store(
            out_ptr,
            index=(row, 0, k),
            tile=ct.reshape(tile2d, (1, CHANNELS, 1)),
        )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="edd7640e", TRANS_ROWS=4)
def oracle_forward(inputs, *, TRANS_ROWS: int):
    arg0_1, *shape_params = inputs
    device = arg0_1.device

    full = torch.empty_strided(
        (1, 1), (1, 1), device=device, dtype=torch.int64
    )
    indices_pad = torch.empty(
        (KWIN_PAD, BLOCK_S), device=device, dtype=torch.int64
    )
    base = torch.empty_strided(
        (ROWS, CHANNELS, KWIN),
        (CHANNELS * KWIN, KWIN, 1),
        device=device,
        dtype=arg0_1.dtype,
    )

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _metadata_kernel, (full, indices_pad))
    ct.launch(stream, (ROWS, 1, 1), _convbert_window_kernel, (arg0_1, base))

    indices = indices_pad[:KWIN].unsqueeze(-1).unsqueeze(-1)
    expand = base.expand(_shape_tuple(shape_params[5]))
    return full, indices, expand, expand.permute(0, 2, 1)
