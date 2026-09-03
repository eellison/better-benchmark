"""cuTile port of sum_456bcf55ec29: Demucs GLU cat + channel-sum.

Two kernels:
1) A producer kernel that computes the concatenated GLU bf16 output and
   also emits per-tile channel partials in fp32.
2) A finalize kernel that reduces the partial-sum tiles into the final
   fp32 [2048] channel sum, with a bf16 rounding boundary at the end.

The Triton oracle used inline_asm add.rn.f32 / mul.rn.f32 and
fp_downcast_rounding="rtne"; these are the default in cuTile so we drop
them.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 1024
C_TOTAL = 2048
TIME = 372
K_TOTAL = BATCH * TIME


@ct.kernel
def _producer_kernel(
    arg0,          # bf16 (BATCH, C_HALF, TIME)
    arg1,          # bf16 (BATCH, C_HALF, TIME)
    gate,          # bf16 (BATCH, C_TOTAL, TIME)
    out,           # bf16 (BATCH, C_TOTAL, TIME)
    partial,       # fp32 (2, NUM_K_TILES, C_HALF)
    NUM_K_TILES: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    # Row (BLOCK_K,) is a linear index over BATCH * TIME.
    k = ct.arange(BLOCK_K, dtype=ct.int32) + k_block * BLOCK_K
    c = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    k_valid = k < K_TOTAL
    c_valid = c < C_HALF
    # (BLOCK_K, BLOCK_C)
    k2 = ct.reshape(k, (BLOCK_K, 1))
    c2 = ct.reshape(c, (1, BLOCK_C))
    k_valid2 = ct.reshape(k_valid, (BLOCK_K, 1))
    c_valid2 = ct.reshape(c_valid, (1, BLOCK_C))
    mask2 = k_valid2 & c_valid2

    batch = k2 // TIME
    time = k2 - batch * TIME

    # For the (batch, c, time) indices, indices need to broadcast to
    # (BLOCK_K, BLOCK_C).
    batch_b = ct.broadcast_to(batch, (BLOCK_K, BLOCK_C))
    time_b = ct.broadcast_to(time, (BLOCK_K, BLOCK_C))
    c_b = ct.broadcast_to(c2, (BLOCK_K, BLOCK_C))
    c_second_b = c_b + C_HALF  # For the gate second half.

    # Gather from arg0, arg1 at (batch, c, time). arg0 is 3D bf16
    # (BATCH, C_HALF, TIME).
    a_bf16 = ct.gather(arg0, (batch_b, c_b, time_b), mask=mask2)
    b_bf16 = ct.gather(arg1, (batch_b, c_b, time_b), mask=mask2)
    a = ct.astype(a_bf16, ct.float32)
    b = ct.astype(b_bf16, ct.float32)
    add_compiled = a + b
    add_eager = ct.astype(ct.astype(add_compiled, ct.bfloat16), ct.float32)

    # Gate is (BATCH, C_TOTAL, TIME).
    first_bf = ct.gather(gate, (batch_b, c_b, time_b), mask=mask2)
    second_bf = ct.gather(gate, (batch_b, c_second_b, time_b), mask=mask2)
    first = ct.astype(first_bf, ct.float32)
    second = ct.astype(second_bf, ct.float32)

    sig = 1.0 / (1.0 + ct.exp(-second))

    out_first_bf = ct.astype(sig * add_compiled, ct.bfloat16)
    out_second_bf = ct.astype(((1.0 - sig) * sig * first) * add_compiled,
                              ct.bfloat16)
    ct.scatter(out, (batch_b, c_b, time_b), out_first_bf, mask=mask2)
    ct.scatter(out, (batch_b, c_second_b, time_b), out_second_bf, mask=mask2)

    sum_first_bf = ct.astype(sig * add_eager, ct.bfloat16)
    sum_second_bf = ct.astype(((1.0 - sig) * sig * first) * add_eager,
                              ct.bfloat16)
    # Partial sums per (k_block, c). Reduce over BLOCK_K axis=0.
    sum_first_f = ct.astype(sum_first_bf, ct.float32)
    sum_second_f = ct.astype(sum_second_bf, ct.float32)
    sum_first_f = ct.where(mask2, sum_first_f, 0.0)
    sum_second_f = ct.where(mask2, sum_second_f, 0.0)
    partial_first = ct.sum(sum_first_f, axis=0)   # (BLOCK_C,)
    partial_second = ct.sum(sum_second_f, axis=0)

    # Store to partial[0, k_block, c] and partial[1, k_block, c].
    ct.scatter(
        partial,
        (
            ct.full((BLOCK_C,), 0, dtype=ct.int32),
            ct.full((BLOCK_C,), k_block, dtype=ct.int32),
            c,
        ),
        partial_first,
        mask=c_valid,
    )
    ct.scatter(
        partial,
        (
            ct.full((BLOCK_C,), 1, dtype=ct.int32),
            ct.full((BLOCK_C,), k_block, dtype=ct.int32),
            c,
        ),
        partial_second,
        mask=c_valid,
    )


@ct.kernel
def _final_kernel(
    partial,        # fp32 (2, NUM_K_TILES, C_HALF)
    reduced,        # fp32 (C_TOTAL,)
    NUM_K_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    c = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    tile_valid = tiles < NUM_K_TILES
    c_valid = c < C_HALF

    tile2 = ct.reshape(tiles, (BLOCK_TILES, 1))
    c2 = ct.reshape(c, (1, BLOCK_C))
    tile_valid2 = ct.reshape(tile_valid, (BLOCK_TILES, 1))
    c_valid2 = ct.reshape(c_valid, (1, BLOCK_C))
    mask2 = tile_valid2 & c_valid2

    tile_b = ct.broadcast_to(tile2, (BLOCK_TILES, BLOCK_C))
    c_b = ct.broadcast_to(c2, (BLOCK_TILES, BLOCK_C))
    zero_b = ct.full((BLOCK_TILES, BLOCK_C), 0, dtype=ct.int32)
    one_b = ct.full((BLOCK_TILES, BLOCK_C), 1, dtype=ct.int32)

    first = ct.gather(partial, (zero_b, tile_b, c_b), mask=mask2)
    second = ct.gather(partial, (one_b, tile_b, c_b), mask=mask2)
    first = ct.where(mask2, first, 0.0)
    second = ct.where(mask2, second, 0.0)

    sum_first = ct.sum(first, axis=0)
    sum_second = ct.sum(second, axis=0)
    # bf16 rounding boundary.
    r_first = ct.astype(ct.astype(sum_first, ct.bfloat16), ct.float32)
    r_second = ct.astype(ct.astype(sum_second, ct.bfloat16), ct.float32)

    ct.scatter(reduced, (c,), r_first, mask=c_valid)
    ct.scatter(reduced, (c + C_HALF,), r_second, mask=c_valid)


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@oracle_impl(hardware="B200", point="a7f82378", BLOCK_K=128, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int):
    arg0, arg1, gate = inputs
    device = arg0.device

    out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    num_k_tiles = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    partial = torch.empty(
        (2, num_k_tiles, C_HALF), device=device, dtype=torch.float32
    )
    reduced = torch.empty((C_TOTAL,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    num_c_blocks = (C_HALF + BLOCK_C - 1) // BLOCK_C
    ct.launch(
        stream,
        (num_c_blocks, num_k_tiles, 1),
        _producer_kernel,
        (arg0, arg1, gate, out, partial, num_k_tiles, BLOCK_K, BLOCK_C),
    )

    block_tiles = _next_pow2(num_k_tiles)
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _final_kernel,
        (partial, reduced, num_k_tiles, block_tiles, BLOCK_C),
    )
    return out, reduced
