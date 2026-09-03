"""cuTile port of pointwise_4c8935159f18: Demucs slice/ReLU/add.

For arg1 of shape (B, C, ARG_TIME), slice[:, :, SLICE_START : SLICE_START+TIME]
and add to relu(arg0). All bf16. TIME = 23212 which is not a power of 2, so
we tile along the last dim with BLOCK and pad; we allocate a padded output and
return a narrowed view. Actually TIME is required output size so we allocate
padded and slice the returned tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 128
TIME = 23212
ARG_TIME = 23923
SLICE_START = 355
ROWS = BATCH * CHANNELS


@ct.kernel
def _slice_relu_add_kernel(
    conv_ptr,   # bf16 [ROWS, TIME_PADDED_conv]
    arg_ptr,    # bf16 [ROWS, ARG_TIME]
    out_ptr,    # bf16 [ROWS, TIME_PADDED_out]
    BLOCK: ct.Constant[int],
    START_TILES: ct.Constant[int],   # SLICE_START // BLOCK
    START_OFFSET: ct.Constant[int],  # SLICE_START % BLOCK
):
    row = ct.bid(0)
    tile = ct.bid(1)
    # For the slice, we need arg[row, SLICE_START + tile*BLOCK : SLICE_START + (tile+1)*BLOCK]
    # SLICE_START = 355. If it doesn't divide BLOCK (355 % 1024 = 355 != 0),
    # we need to load two adjacent tiles and shift. Complex.
    # Simpler approach: use scalar-shifted loads. But cuTile tile indices must
    # be compile-time simple. We need to load arg[row, SLICE_START + tile*BLOCK:...]
    # which via tile-space is at tile-index (row, (SLICE_START + tile*BLOCK)/BLOCK)
    # if SLICE_START is a multiple of BLOCK. Since it isn't, we load one aligned tile
    # then use a shift + concat. Or: just do it in scalars using load_advanced_indexing.
    #
    # Given the constraint, prefer a simpler implementation: use a stride-1 flat
    # index into the input via a linear address. cuTile's `Array` supports
    # arbitrary strides so `arg_ptr.view(ROWS, ARG_TIME)` works fine, but the
    # `index=` is tile-space multiplied by the tile shape.
    #
    # Trick: pass in a shifted view of arg. We can as_strided the arg tensor
    # in Python so that from cuTile's view, the base is [ROWS, TIME_PAD] starting
    # at SLICE_START in the last dim. This does not change strides (last is 1);
    # only the storage offset moves. cuTile respects that.
    conv = ct.load(conv_ptr, index=(row, tile), shape=(1, BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    sliced = ct.load(arg_ptr, index=(row, tile), shape=(1, BLOCK),
                     padding_mode=ct.PaddingMode.ZERO)
    # relu preserving NaN
    conv_f = ct.astype(conv, ct.float32)
    zeros = ct.full(shape=(1, BLOCK), fill_value=0.0, dtype=ct.float32)
    isnan = conv_f != conv_f
    relu = ct.where(isnan, conv_f, ct.maximum(conv_f, zeros))
    # add
    add = relu + ct.astype(sliced, ct.float32)
    ct.store(out_ptr, index=(row, tile), tile=ct.astype(add, ct.bfloat16))


@oracle_impl(hardware="B200", point="a3ff9591", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    conv, arg = inputs
    # Ensure BLOCK-aligned output size
    padded_time = ((TIME + BLOCK - 1) // BLOCK) * BLOCK
    out_padded = torch.empty_strided(
        (BATCH, CHANNELS, padded_time),
        (CHANNELS * padded_time, padded_time, 1),
        device=conv.device, dtype=torch.bfloat16,
    )
    # Flatten to (ROWS, padded_time). We view conv as (ROWS, TIME) which will
    # cause OOB loads padded to zero. The arg tensor is shifted so we look
    # from SLICE_START.
    conv_flat = conv.view(ROWS, TIME)
    # Shift arg so column 0 in kernel corresponds to arg[..., SLICE_START].
    # Use as_strided keeping storage offset SLICE_START.
    arg_shifted = torch.as_strided(
        arg, (ROWS, ARG_TIME - SLICE_START),
        (ARG_TIME, 1), storage_offset=SLICE_START,
    )
    out_flat = out_padded.view(ROWS, padded_time)
    stream = torch.cuda.current_stream()
    grid = (ROWS, padded_time // BLOCK, 1)
    ct.launch(stream, grid, _slice_relu_add_kernel,
              (conv_flat, arg_shifted, out_flat, BLOCK, 0, 0))
    return out_padded[:, :, :TIME].contiguous()
