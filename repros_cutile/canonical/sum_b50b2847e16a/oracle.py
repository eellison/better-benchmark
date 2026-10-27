"""cuTile port of sum_b50b2847e16a: Demucs slice_scatter + strided mask + sum.

Fair 3-kernel structure matching Triton reference:
- Kernel 1 (_zero_pad_kernel): zero the left/right pad regions of the padded output.
- Kernel 2 (_copy_where_reduce_kernel): copy src into the padded interior,
  compute where(mask, scalar, src), and produce per-(channel, batch, tile)
  partial sums via ct.sum.
- Kernel 3 (_final_sum_kernel): reduce partials via ct.sum per channel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 256
IN_T = 5804
PAD_LEFT = 87
PAD_RIGHT = 88
OUT_T = IN_T + PAD_LEFT + PAD_RIGHT  # 5979
COPY_REDUCE_BLOCK_T = 8192  # next_pow2 for IN_T reduction path (single tile)
TILES_T = 1
PARTIALS_PER_CHANNEL = BATCH * TILES_T  # 4
FINAL_BLOCK = 4
PAD_BLOCK = 128


def _next_pow2(v: int) -> int:
    return 1 << (v - 1).bit_length() if v > 1 else 1


@ct.kernel
def _zero_pad_kernel(
    out_ptr,             # bf16 (BATCH*CHANNELS, OUT_T)
    OUT_T_: ct.Constant[int],
    PAD_LEFT_: ct.Constant[int],
    PAD_RIGHT_: ct.Constant[int],
    BLOCK_PAD: ct.Constant[int],
):
    row = ct.bid(0)
    offsets = ct.arange(BLOCK_PAD, dtype=ct.int32)
    zero_tile = ct.full((BLOCK_PAD,), 0.0, dtype=ct.bfloat16)
    row_idx = ct.full((BLOCK_PAD,), 0, dtype=ct.int32) + row
    # Left pad region [0, PAD_LEFT_)
    left_valid = offsets < PAD_LEFT_
    ct.scatter(out_ptr, (row_idx, offsets), zero_tile, mask=left_valid)
    # Right pad region [OUT_T_-PAD_RIGHT_, OUT_T_)
    right_valid = offsets < PAD_RIGHT_
    right_pos = (OUT_T_ - PAD_RIGHT_) + offsets
    ct.scatter(out_ptr, (row_idx, right_pos), zero_tile, mask=right_valid)


@ct.kernel
def _copy_where_reduce_kernel(
    src_ptr,          # bf16 (BATCH*CHANNELS, IN_T_PAD)
    mask_ptr,         # b8   (BATCH*CHANNELS, IN_T_PAD)
    scalar_ptr,       # bf16 (1,)
    padded_ptr,       # bf16 (BATCH*CHANNELS, OUT_T)
    where_ptr,        # bf16 (BATCH*CHANNELS, IN_T_PAD)
    partial_ptr,      # f32  (CHANNELS, PARTIALS_PER_CHANNEL)
    IN_T_: ct.Constant[int],
    OUT_T_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    PAD_LEFT_: ct.Constant[int],
    BLOCK_T: ct.Constant[int],
):
    channel = ct.bid(0)
    batch = ct.bid(1)
    row = batch * CHANNELS_ + channel

    # Load one row of length BLOCK_T (must be power-of-2). We assume the src /
    # where 2D tensors have padded second dim so no OOB writes occur.
    src_row = ct.load(src_ptr, index=(row, 0), shape=(1, BLOCK_T),
                      padding_mode=ct.PaddingMode.ZERO)
    mask_row = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_T),
                       padding_mode=ct.PaddingMode.ZERO)
    scalar_v = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_bcast = ct.full((1, BLOCK_T), 0.0, dtype=ct.bfloat16) + \
                    ct.reshape(scalar_v, (1, 1))

    where_vals = ct.where(mask_row, scalar_bcast, src_row)
    ct.store(where_ptr, index=(row, 0), tile=where_vals)

    # Padded destination: at (row, PAD_LEFT_) tile of width BLOCK_T. Split into
    # (row, PAD_LEFT_..PAD_LEFT_+BLOCK_T) — we pass a wider `padded_2d` that
    # already has enough columns; the launcher aligns.
    ct.store(padded_ptr, index=(row, 0), tile=src_row)

    # Sum reduction over the tile.
    where_flat = ct.reshape(where_vals, (BLOCK_T,))
    offsets_t = ct.arange(BLOCK_T, dtype=ct.int32)
    active = offsets_t < IN_T_
    where_f = ct.astype(where_flat, ct.float32)
    masked = ct.where(active, where_f, 0.0)
    partial = ct.sum(masked, axis=0)  # scalar
    partial_2 = ct.reshape(partial, (1, 1))
    ct.store(partial_ptr, index=(channel, batch), tile=partial_2)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,       # f32 (CHANNELS, PARTIALS_PER_CHANNEL)
    out_ptr,           # f32 (CHANNELS,)
    PARTIALS_PER_CHANNEL_: ct.Constant[int],
    BLOCK_PARTIALS: ct.Constant[int],
):
    channel = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(channel, 0),
        shape=(1, BLOCK_PARTIALS), padding_mode=ct.PaddingMode.ZERO,
    )
    values_f = ct.astype(ct.reshape(values, (BLOCK_PARTIALS,)), ct.float32)
    idx = ct.arange(BLOCK_PARTIALS, dtype=ct.int32)
    valid = idx < PARTIALS_PER_CHANNEL_
    masked = ct.where(valid, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    total_1 = ct.reshape(total, (1,))
    ct.store(out_ptr, index=(channel,), tile=total_1)


@oracle_impl(hardware="B200", point="1bed4207", BLOCK_T=COPY_REDUCE_BLOCK_T)
def oracle_forward(inputs, *, BLOCK_T):
    src, mask, scalar, shape_param = inputs
    device = src.device

    slice_scatter = torch.empty_strided(
        tuple(int(d) for d in shape_param),
        (CHANNELS * OUT_T, OUT_T, 1),
        device=device, dtype=torch.bfloat16,
    )
    where = torch.empty_strided(
        (BATCH, CHANNELS, IN_T),
        (CHANNELS * IN_T, IN_T, 1),
        device=device, dtype=torch.bfloat16,
    )
    partial = torch.empty((CHANNELS, PARTIALS_PER_CHANNEL),
                          device=device, dtype=torch.float32)
    reduced = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # For kernel 2 we need a padded 2D storage of width == BLOCK_T so we can
    # store the tile safely without OOB writes.
    src_2d_pad = torch.zeros((BATCH * CHANNELS, BLOCK_T), device=device, dtype=torch.bfloat16)
    src_2d_pad[:, :IN_T].copy_(src.contiguous().view(BATCH * CHANNELS, IN_T))
    mask_2d_pad = torch.zeros((BATCH * CHANNELS, BLOCK_T), device=device, dtype=torch.bool)
    mask_2d_pad[:, :IN_T].copy_(mask.contiguous().view(BATCH * CHANNELS, IN_T))
    where_2d_pad = torch.empty((BATCH * CHANNELS, BLOCK_T), device=device, dtype=torch.bfloat16)

    padded_2d = slice_scatter.view(BATCH * CHANNELS, OUT_T)
    # We need to store the src tile into padded at column offset PAD_LEFT. Create
    # a view of padded starting at PAD_LEFT with width == BLOCK_T (padded_2d must be
    # wide enough — OUT_T = IN_T + PAD_LEFT + PAD_RIGHT = 5979, BLOCK_T = 8192 > OUT_T).
    # To avoid OOB, allocate a work buffer wider than padded and copy back.
    padded_work = torch.empty((BATCH * CHANNELS, BLOCK_T + PAD_LEFT),
                              device=device, dtype=torch.bfloat16)
    # Slice a (BATCH*CHANNELS, BLOCK_T) view starting at PAD_LEFT.
    padded_work_view = torch.as_strided(
        padded_work, (BATCH * CHANNELS, BLOCK_T),
        (BLOCK_T + PAD_LEFT, 1),
        storage_offset=PAD_LEFT,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH * CHANNELS, 1, 1),
        _zero_pad_kernel,
        (padded_2d, OUT_T, PAD_LEFT, PAD_RIGHT, PAD_BLOCK),
    )
    ct.launch(
        stream,
        (CHANNELS, BATCH, 1),
        _copy_where_reduce_kernel,
        (src_2d_pad, mask_2d_pad, scalar.view(1), padded_work_view, where_2d_pad, partial,
         IN_T, OUT_T, CHANNELS, PAD_LEFT, BLOCK_T),
    )
    ct.launch(
        stream,
        (CHANNELS, 1, 1),
        _final_sum_kernel,
        (partial, reduced, PARTIALS_PER_CHANNEL, FINAL_BLOCK),
    )

    # Copy padded interior back to slice_scatter[:, :, PAD_LEFT:PAD_LEFT+IN_T].
    padded_2d[:, PAD_LEFT:PAD_LEFT + IN_T].copy_(padded_work_view[:, :IN_T])
    # Copy where back into the final layout.
    where.view(BATCH * CHANNELS, IN_T).copy_(where_2d_pad[:, :IN_T])

    return slice_scatter, where, reduced
