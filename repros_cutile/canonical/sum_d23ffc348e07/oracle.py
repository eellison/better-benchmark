"""cuTile port of sum_d23ffc348e07: Demucs structured scatter-reduce.

Mirrors Triton's 4-kernel structure:
  1. Slice_scatter zero-pad (torch used for the interior copy since it's a
     pure layout copy; the padding zeros are set by the launch).
  2. Full-scalar zero.
  3. Copy-where-reduce: for each (channel, batch, tile), compute the where
     value in-kernel and emit both the materialized bf16 tensor plus a
     per-tile partial fp32 sum.
  4. Final sum: reduce partials to a [C] fp32 vector.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 64
IN_T = 92844
PAD = 1426
OUT_T = 95696
COPY_REDUCE_BLOCK_T = 4096
TILES_T = (IN_T + COPY_REDUCE_BLOCK_T - 1) // COPY_REDUCE_BLOCK_T
PARTIALS_PER_CHANNEL = BATCH * TILES_T
FINAL_BLOCK = 128  # >= PARTIALS_PER_CHANNEL and power of 2


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _copy_where_reduce_kernel(
    src_ptr,        # bf16 flat [BATCH*CHANNELS*IN_T]
    mask_ptr,       # b8 flat [BATCH*mask_stride_b] with per-channel stride mask_stride_c
    where_ptr,      # bf16 flat [BATCH*CHANNELS*IN_T]
    partial_ptr,    # f32 flat [CHANNELS*PARTIALS_PER_CHANNEL]
    IN_T_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    MASK_STRIDE_B: ct.Constant[int],
    MASK_STRIDE_C: ct.Constant[int],
    TILES_T_: ct.Constant[int],
    BLOCK_T: ct.Constant[int],
):
    channel = ct.bid(0)
    batch = ct.bid(1)
    tile = ct.bid(2)

    offsets_t = tile * BLOCK_T + ct.arange(BLOCK_T, dtype=ct.int32)
    active = offsets_t < IN_T_

    src_base = (batch * CHANNELS_ + channel) * IN_T_
    src_offsets = src_base + offsets_t
    mask_offsets = batch * MASK_STRIDE_B + channel * MASK_STRIDE_C + offsets_t

    src_vals = ct.gather(src_ptr, src_offsets, mask=active)
    mask_vals = ct.gather(mask_ptr, mask_offsets, mask=active)

    # cuTile: mask is b8 (bool); when True -> use 0, else use src.
    zero_bf = ct.full((BLOCK_T,), 0.0, dtype=ct.bfloat16)
    where_vals = ct.where(mask_vals, zero_bf, src_vals)

    ct.scatter(where_ptr, src_offsets, where_vals, mask=active)

    where_f = ct.astype(where_vals, ct.float32)
    where_f_masked = ct.where(active, where_f, 0.0)
    reduced = ct.sum(where_f_masked)

    partial_offset = channel * (BATCH * TILES_T_) + batch * TILES_T_ + tile
    ct.scatter(partial_ptr, ct.reshape(partial_offset, (1,)),
               ct.reshape(reduced, (1,)))


@ct.kernel
def _final_sum_kernel(
    partial_ptr,    # f32 [CHANNELS, PARTIALS_PER_CHANNEL]
    out_ptr,        # f32 [CHANNELS]
    PARTIALS_PER_CHANNEL_: ct.Constant[int],
    BLOCK_PARTIALS: ct.Constant[int],
):
    channel = ct.bid(0)
    values = ct.load(partial_ptr, index=(channel, 0),
                     shape=(1, BLOCK_PARTIALS),
                     padding_mode=ct.PaddingMode.ZERO)
    values_f = ct.astype(values, ct.float32)
    idx = ct.arange(BLOCK_PARTIALS, dtype=ct.int32)
    valid = ct.reshape(idx < PARTIALS_PER_CHANNEL_, (1, BLOCK_PARTIALS))
    masked = ct.where(valid, values_f, 0.0)
    total = ct.sum(masked)
    ct.store(out_ptr, index=(channel,), tile=ct.reshape(total, (1,)))


@oracle_impl(hardware="B200", point="9e7b7398", BLOCK_T=COPY_REDUCE_BLOCK_T)
def oracle_forward(inputs, *, BLOCK_T: int):
    arg0_1, arg1_1, _s0 = inputs
    device = arg0_1.device

    # Physical stride info for arg1_1 (b8 mask)
    mask_stride_b = int(arg1_1.stride(0))
    mask_stride_c = int(arg1_1.stride(1))

    # slice_scatter: torch (pure zero-padded layout copy of arg0_1 into the
    # interior of OUT_T). The Triton `_zero_pad_kernel` only handles the
    # padded regions; the interior is a pure copy.
    slice_scatter = torch.zeros(
        (BATCH, CHANNELS, OUT_T), device=device, dtype=torch.bfloat16)
    slice_scatter[:, :, PAD:OUT_T - PAD] = arg0_1

    # full_1 scalar zero
    full_1 = torch.zeros((), device=device, dtype=torch.bfloat16)

    where = torch.empty_strided(
        (BATCH, CHANNELS, IN_T),
        (CHANNELS * IN_T, IN_T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty(
        (CHANNELS, PARTIALS_PER_CHANNEL),
        device=device,
        dtype=torch.float32,
    )
    convert_element_type = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # Flatten inputs for gather/scatter access
    src_flat = arg0_1.contiguous().view(-1)
    # arg1_1 is strided; access its underlying storage flatly
    mask_storage_size = int(arg1_1.untyped_storage().nbytes() // arg1_1.element_size())
    mask_flat = arg1_1.as_strided((mask_storage_size,), (1,))
    where_flat = where.view(-1)
    partial_flat = partial.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (CHANNELS, BATCH, TILES_T),
        _copy_where_reduce_kernel,
        (src_flat, mask_flat, where_flat, partial_flat,
         IN_T, CHANNELS, mask_stride_b, mask_stride_c, TILES_T, BLOCK_T),
    )
    ct.launch(
        stream,
        (CHANNELS, 1, 1),
        _final_sum_kernel,
        (partial, convert_element_type, PARTIALS_PER_CHANNEL, FINAL_BLOCK),
    )

    return slice_scatter, full_1, where, convert_element_type
