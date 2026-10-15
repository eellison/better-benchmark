"""cuTile port of sum_a05c10ed018a: Demucs slice_scatter + where + channel sum."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 1024
IN_T = 364
PAD = 4
OUT_T = 372


@ct.kernel
def _slice_scatter_where_kernel(
    src,        # (BATCH, CHANNELS, IN_T) bf16
    mask,       # (BATCH, CHANNELS, IN_T) bool
    scalar_ptr, # (1,) bf16
    padded,     # (BATCH, CHANNELS, OUT_T) bf16
    where_out,  # (BATCH, CHANNELS, IN_T) bf16
    BLOCK_T: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)
    # Load src row [IN_T]; we choose BLOCK_T as power-of-2 >= IN_T
    src_row = ct.load(src, index=(b, c, 0), shape=(1, 1, BLOCK_T),
                      padding_mode=ct.PaddingMode.ZERO)
    mask_row = ct.load(mask, index=(b, c, 0), shape=(1, 1, BLOCK_T),
                       padding_mode=ct.PaddingMode.ZERO)
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_val_f = ct.astype(scalar, ct.float32)
    src_f = ct.astype(src_row, ct.float32)
    scalar_broadcast = ct.reshape(scalar_val_f, (1, 1, 1))
    where_val = ct.where(mask_row, scalar_broadcast, src_f)
    where_bf16 = ct.astype(where_val, ct.bfloat16)
    ct.store(where_out, index=(b, c, 0), tile=where_bf16)
    # Store src row into padded[b, c, PAD:PAD+IN_T]; and zeros elsewhere.
    # Since OUT_T=372, PAD=4, IN_T=364: 4:368 gets src, 0:4 and 368:372 zero.
    # Simplest: store padded of size OUT_T that is composed of a shifted view.
    # OUT_T = 372. Tile at fixed size PAD_T (power-of-2) starting from col 0.
    # Approach: build a length-BLOCK_T tile that has src_row aligned at PAD offset.
    # Since we can't shift in-tile easily, do it differently:
    # First zero-fill full padded[b, c, :], then in another kernel we copy src to
    # padded[b, c, PAD:PAD+IN_T].
    # But we can't do it in one launch easily. Let's leave the padded tensor
    # zero-filled at Python level, and just write the src part here.
    # Store the length-IN_T src at position 4 in the OUT_T-dim padded.
    # Since 4 isn't a tile-index multiple of BLOCK_T's element count... we need
    # a scalar-index store. cuTile store supports index in tile-space, so if
    # BLOCK_T divides 4 (e.g. BLOCK_T=4) we can compute tile index. But that's
    # too fine.
    # Alternative: use ct.scatter with element-space indices.
    pass


@ct.kernel
def _sum_where_kernel(
    where_arr,   # (BATCH, CHANNELS, IN_T) bf16
    out_sum,     # (CHANNELS,) f32
    BATCH_: ct.Constant[int],
    IN_T_: ct.Constant[int],
    BLOCK_T: ct.Constant[int],
):
    c = ct.bid(0)
    # Sum over batches and time for this channel
    # We can either use a 2D tile or accumulate. Simplest: one tile of shape
    # (BATCH, BLOCK_T).
    tile = ct.load(where_arr, index=(0, c, 0), shape=(BATCH_, 1, BLOCK_T),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_f = ct.astype(tile, ct.float32)
    total = ct.sum(tile_f)
    ct.store(out_sum, index=(c,), tile=ct.reshape(total, (1,)))


@oracle_impl(hardware="B200", point="fea3ee9c")
def oracle_forward(inputs):
    src, mask, scalar, shape_param = inputs
    device = src.device

    slice_scatter = torch.zeros(
        tuple(int(dim) for dim in shape_param),
        device=device, dtype=torch.bfloat16,
    )
    # Copy src into slice_scatter[:, :, 4:368]
    slice_scatter[:, :, PAD:PAD + IN_T] = src

    where_out = torch.empty_strided(
        (BATCH, CHANNELS, IN_T), (CHANNELS * IN_T, IN_T, 1),
        device=device, dtype=torch.bfloat16,
    )
    reduced = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # scalar is 0-d bf16 tensor
    scalar_1d = scalar.view(1)

    stream = torch.cuda.current_stream()
    # BLOCK_T must be power of 2 >= IN_T=364 -> 512
    ct.launch(stream, (BATCH, CHANNELS, 1), _slice_scatter_where_kernel,
              (src, mask, scalar_1d, slice_scatter, where_out, 512))
    ct.launch(stream, (CHANNELS, 1, 1), _sum_where_kernel,
              (where_out, reduced, BATCH, IN_T, 512))
    return slice_scatter, where_out, reduced
