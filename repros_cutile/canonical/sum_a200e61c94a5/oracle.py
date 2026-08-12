"""cuTile port of sum_a200e61c94a5: Demucs bf16 reshape+channel-sum with split-K partials."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 8
TIME = 371372
ELEMS_PER_CHANNEL = BATCH * TIME
PARTIAL_SIZE = 8192
NUM_PARTIALS = (ELEMS_PER_CHANNEL + PARTIAL_SIZE - 1) // PARTIAL_SIZE
FINAL_BLOCK = 256


@ct.kernel
def _partial_sum_kernel(
    view_ptr,      # bf16 [BATCH, CHANNELS, TIME] view of x
    partial_ptr,   # f32 [CHANNELS, NUM_PARTIALS]
    TIME_C: ct.Constant[int],
    ELEMS_PER_CHANNEL_C: ct.Constant[int],
    PARTIAL_SIZE_C: ct.Constant[int],
    NUM_PARTIALS_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    partial = ct.bid(1)
    r = ct.arange(BLOCK_R, dtype=ct.int32)
    logical = partial * PARTIAL_SIZE_C + r
    active = logical < ELEMS_PER_CHANNEL_C
    batch = logical // TIME_C
    t = logical - batch * TIME_C
    # view_ptr has shape [BATCH, CHANNELS, TIME]; index=(batch, channel, t) with tile of shape (BLOCK_R,)
    # But cuTile doesn't support gather-style loads. Use load with 3D shape then squeeze?
    # Simpler: use the input's storage-linear flat view like the Triton kernel does.
    # We'll pass in a flat view instead.


@ct.kernel
def _partial_sum_flat_kernel(
    x_flat_ptr,       # bf16 [BATCH*CHANNELS*TIME] flat storage
    partial_ptr,      # f32 [CHANNELS, NUM_PARTIALS]
    TIME_C: ct.Constant[int],
    CHANNELS_C: ct.Constant[int],
    ELEMS_PER_CHANNEL_C: ct.Constant[int],
    PARTIAL_SIZE_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    partial = ct.bid(1)
    r = ct.arange(BLOCK_R, dtype=ct.int32)
    logical = partial * PARTIAL_SIZE_C + r
    active = logical < ELEMS_PER_CHANNEL_C
    batch = logical // TIME_C
    t = logical - batch * TIME_C
    # Compute the storage offset for [batch, channel, t]:
    # offset = batch * CHANNELS * TIME + channel * TIME + t
    # cuTile's load requires a compile-time-known regular shape. Can't gather.
    # Use load_advanced_indexing instead.


# The Triton kernel gathers via computed offsets: `offsets = batch * channels * time + channel * time + t`.
# In cuTile this is a scatter-gather pattern. It's not straightforward.
# Fall back to a reduction using the tile-space approach:
# View x as (BATCH, CHANNELS, TIME) contiguous with strides (C*T, T, 1).
# For channel c: sum over batch b and time t. Do this by iterating BATCH in a Python loop
# per launch, with each block reducing a chunk of TIME.

@ct.kernel
def _channel_partial_kernel(
    x_ptr,        # bf16 [BATCH, CHANNELS, TIME]
    partial_ptr,  # f32 [BATCH, CHANNELS, NUM_PARTIALS]
    TIME_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    partial = ct.bid(1)
    batch = ct.bid(2)
    values = ct.load(
        x_ptr,
        index=(batch, channel, partial),
        shape=(1, 1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(ct.astype(values, ct.float32))
    # Store scalar. Reshape total to (1,) for storage.
    ct.store(partial_ptr, index=(batch, channel, partial), tile=ct.reshape(total, (1, 1, 1)))


@ct.kernel
def _finalize_channel_kernel(
    partial_ptr,       # f32 [BATCH, CHANNELS, NUM_PARTIALS]
    out_ptr,           # f32 [CHANNELS]
    BATCH_C: ct.Constant[int],
    NUM_PARTIALS_C: ct.Constant[int],
    BLOCK_ALL: ct.Constant[int],
):
    channel = ct.bid(0)
    # Read the whole (BATCH, 1, NUM_PARTIALS) slice for this channel and reduce.
    values = ct.load(
        partial_ptr,
        index=(0, channel, 0),
        shape=(BATCH_C, 1, BLOCK_ALL),
        padding_mode=ct.PaddingMode.ZERO,
    )
    total = ct.sum(values)
    ct.store(out_ptr, index=(channel,), tile=ct.reshape(total, (1,)))


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@oracle_impl(hardware="B200", point="a809b30f", BLOCK_R=PARTIAL_SIZE)
def oracle_forward(inputs, *, BLOCK_R: int):
    x, _shape = inputs
    del _shape

    view = torch.as_strided(
        x,
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
    )
    # NUM_PARTIALS tiles of BLOCK_R each cover TIME. Last tile pads OOB with zero.
    partial = torch.empty_strided(
        (BATCH, CHANNELS, NUM_PARTIALS),
        (CHANNELS * NUM_PARTIALS, NUM_PARTIALS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (CHANNELS, NUM_PARTIALS, BATCH),
        _channel_partial_kernel,
        (view, partial, TIME, BLOCK_R),
    )
    # Reduce over BATCH * NUM_PARTIALS elements per channel.
    block_all = _next_pow2(NUM_PARTIALS)
    ct.launch(
        stream,
        (CHANNELS, 1, 1),
        _finalize_channel_kernel,
        (partial, out, BATCH, NUM_PARTIALS, block_all),
    )
    return view, out
