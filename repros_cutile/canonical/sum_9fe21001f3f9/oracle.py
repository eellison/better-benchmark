"""cuTile port of sum_9fe21001f3f9: Demucs where + per-channel sum.

Materializes where(mask <= 0, scalar, source) for the visible output and
accumulates the per-channel sum via a T-tile loop that emits partials, then
a small kernel finalizes the sum over batch and T-tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _where_partial_kernel(
    mask_ptr,       # bf16 [B, C, T]
    scalar_ptr,     # bf16 (1,)
    source_ptr,     # bf16 [B, C, T]
    where_ptr,      # bf16 [B, C, T]
    partial_ptr,    # f32 [B, C, TILES]
    T: ct.Constant[int],
    T_TILE: ct.Constant[int],
    TILES: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)
    t_block = ct.bid(2)

    mask = ct.load(mask_ptr, index=(b, c, t_block), shape=(1, 1, T_TILE),
                   padding_mode=ct.PaddingMode.ZERO)
    src = ct.load(source_ptr, index=(b, c, t_block), shape=(1, 1, T_TILE),
                  padding_mode=ct.PaddingMode.ZERO)
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))

    zero_bf = ct.full(shape=(1, 1, T_TILE), fill_value=0.0, dtype=ct.bfloat16)
    take_scalar = mask <= zero_bf
    scalar_3d = ct.reshape(scalar, (1, 1, 1))
    scalar_bcast = ct.full(shape=(1, 1, T_TILE), fill_value=0.0, dtype=ct.bfloat16) + scalar_3d
    values = ct.where(take_scalar, scalar_bcast, src)
    ct.store(where_ptr, index=(b, c, t_block), tile=values)

    # Mask out OOB positions to preserve correct sum.
    t_offsets = ct.arange(T_TILE, dtype=ct.int32) + t_block * T_TILE
    t_offsets_3d = ct.reshape(t_offsets, (1, 1, T_TILE))
    valid = t_offsets_3d < ct.full(shape=(1, 1, T_TILE), fill_value=T, dtype=ct.int32)
    zero_f32 = ct.full(shape=(1, 1, T_TILE), fill_value=0.0, dtype=ct.float32)
    values_f32 = ct.where(valid, ct.astype(values, ct.float32), zero_f32)
    partial = ct.sum(values_f32)
    ct.store(partial_ptr, index=(b, c, t_block),
             tile=ct.full(shape=(1, 1, 1), fill_value=partial, dtype=ct.float32))


@ct.kernel
def _finalize_kernel(
    partial_ptr,    # f32 [B, C, TILES]
    out_ptr,        # f32 [C]
    B: ct.Constant[int],
    TILES: ct.Constant[int],
    B_TILES: ct.Constant[int],
    ACCUM_TILE: ct.Constant[int],
):
    c = ct.bid(0)
    # Sum partial[b, c, t] over (b, t)
    # Load a slab (B, TILES) at column c.
    slab = ct.load(partial_ptr, index=(0, c, 0), shape=(B_TILES, 1, ACCUM_TILE),
                   padding_mode=ct.PaddingMode.ZERO)
    # Mask off OOB batches / tiles.
    bs = ct.arange(B_TILES, dtype=ct.int32)
    ts = ct.arange(ACCUM_TILE, dtype=ct.int32)
    bs_3d = ct.reshape(bs, (B_TILES, 1, 1))
    ts_3d = ct.reshape(ts, (1, 1, ACCUM_TILE))
    valid_b = bs_3d < ct.full(shape=(B_TILES, 1, 1), fill_value=B, dtype=ct.int32)
    valid_t = ts_3d < ct.full(shape=(1, 1, ACCUM_TILE), fill_value=TILES, dtype=ct.int32)
    valid = valid_b & valid_t
    zero = ct.full(shape=(B_TILES, 1, ACCUM_TILE), fill_value=0.0, dtype=ct.float32)
    masked = ct.where(valid, slab, zero)
    total = ct.sum(masked)
    ct.store(out_ptr, index=(c,),
             tile=ct.full(shape=(1,), fill_value=total, dtype=ct.float32))


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _launch(inputs):
    mask_input, scalar, source = inputs
    b = int(mask_input.shape[0])
    c = int(mask_input.shape[1])
    t = int(mask_input.shape[2])
    device = mask_input.device

    where = torch.empty_strided(
        (b, c, t),
        (c * t, t, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)

    # Pick a T tile size ~1024 to minimize partial tensor size.
    # We want t_tile as small as possible while pow2. Cap at t (rounded up).
    t_tile = 1024
    while t_tile > t and t_tile > 1:
        t_tile //= 2
    # Ensure pow2
    if t_tile < 1:
        t_tile = 1
    tiles = (t + t_tile - 1) // t_tile

    partial = torch.empty((b, c, tiles), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (b, c, tiles),
        _where_partial_kernel,
        (mask_input, scalar.view(1), source, where, partial, t, t_tile, tiles),
    )

    b_tiles = _next_pow2(b)
    accum_tile = _next_pow2(tiles)
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_kernel,
        (partial, out, b, tiles, b_tiles, accum_tile),
    )
    return where, out


@oracle_impl(hardware="B200", point="7865ae61")
@oracle_impl(hardware="B200", point="c6e644f9")
@oracle_impl(hardware="B200", point="9d21ed2c")
@oracle_impl(hardware="B200", point="03da249b")
@oracle_impl(hardware="B200", point="11e513cb")
def oracle_forward(inputs):
    return _launch(inputs)
