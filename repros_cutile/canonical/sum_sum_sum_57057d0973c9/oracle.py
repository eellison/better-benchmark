"""cuTile port of sum_sum_sum_57057d0973c9: BEiT 12-branch relative-position scatter-reduce.

For each of the 12 (x, idx) pairs, we:
1. Reduce over the batch axis to get a bf16 [12,197,197] per-channel-per-spatial
   value (rounded via bf16 boundary).
2. Scatter-add into f32[732,12] outputs at buckets given by idx[h,w].

Since each of the 12 branches is independent, we launch 12 identical kernels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
CHANNELS = 12
HEIGHT = 197
WIDTH = 197
SPATIAL = HEIGHT * WIDTH
BUCKETS = 732


@ct.kernel
def _scatter_branch_kernel(
    x,           # bf16 (N, CHANNELS, HEIGHT, WIDTH)
    idx,         # i64 (HEIGHT, WIDTH)
    out,         # fp32 (BUCKETS, CHANNELS)
    N_ITEMS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    channel = ct.bid(0)
    spatial_block = ct.bid(1)

    s_offsets = ct.arange(BLOCK_S, dtype=ct.int32) + spatial_block * BLOCK_S
    s_valid = s_offsets < SPATIAL
    h = s_offsets // WIDTH
    w = s_offsets - h * WIDTH

    # Accumulate the batch reduction into an fp32 (BLOCK_S,) tile.
    partial = ct.zeros((BLOCK_S,), dtype=ct.float32)
    # Loop over batch in tiles of BLOCK_N.
    for n_base in range(0, N_ITEMS, BLOCK_N):
        # Build 2D indices (BLOCK_N, BLOCK_S) into x[n, c, h, w].
        n_offsets = ct.arange(BLOCK_N, dtype=ct.int32) + n_base
        n_valid = n_offsets < N_ITEMS
        # (BLOCK_N, BLOCK_S) tiles for gather
        n2 = ct.reshape(n_offsets, (BLOCK_N, 1))
        h2 = ct.reshape(h, (1, BLOCK_S))
        w2 = ct.reshape(w, (1, BLOCK_S))
        n_valid2 = ct.reshape(n_valid, (BLOCK_N, 1))
        s_valid2 = ct.reshape(s_valid, (1, BLOCK_S))
        mask2 = n_valid2 & s_valid2

        # For 4D array x, gather indices (n, c, h, w). c is scalar (channel).
        n_b = ct.broadcast_to(n2, (BLOCK_N, BLOCK_S))
        h_b = ct.broadcast_to(h2, (BLOCK_N, BLOCK_S))
        w_b = ct.broadcast_to(w2, (BLOCK_N, BLOCK_S))
        c_b = ct.full((BLOCK_N, BLOCK_S), channel, dtype=ct.int32)
        vals_bf16 = ct.gather(x, (n_b, c_b, h_b, w_b), mask=mask2)
        vals_f = ct.astype(vals_bf16, ct.float32)
        vals_f = ct.where(mask2, vals_f, 0.0)
        partial = partial + ct.sum(vals_f, axis=0)

    # Round via bf16 to match the semantics.
    rounded = ct.astype(ct.astype(partial, ct.bfloat16), ct.float32)

    # Fetch bucket for each spatial position: idx is (HEIGHT, WIDTH) i64.
    bucket = ct.gather(idx, (h, w), mask=s_valid)
    bucket_i32 = ct.astype(bucket, ct.int32)

    # atomic_add(out[bucket, channel] += rounded[s]) for each s.
    # out is (BUCKETS, CHANNELS). Use 1D indices tile of shape (BLOCK_S,).
    ch_tile = ct.full((BLOCK_S,), channel, dtype=ct.int32)
    # We need to mask out-of-bounds spatial positions. atomic_add with
    # check_bounds=True will drop atomics where bucket is out of bounds,
    # but we also need to skip s_offsets >= SPATIAL. Do this by setting
    # bucket to a large invalid value.
    invalid_bucket = ct.full((BLOCK_S,), BUCKETS, dtype=ct.int32)  # OOB
    bucket_safe = ct.where(s_valid, bucket_i32, invalid_bucket)
    ct.atomic_add(out, (bucket_safe, ch_tile), rounded)


def _launch_branch(x, idx, out, *, block_n, block_s):
    stream = torch.cuda.current_stream()
    grid = (CHANNELS, (SPATIAL + block_s - 1) // block_s, 1)
    ct.launch(
        stream,
        grid,
        _scatter_branch_kernel,
        (x, idx, out, N, block_n, block_s),
    )


@oracle_impl(
    hardware="B200",
    point="019a4c87",
    BLOCK_N=4,
    BLOCK_S=256,
    ZERO_BLOCK=1024,
    scatter_warps=4,
    zero_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    BLOCK_S: int,
    ZERO_BLOCK: int,
    scatter_warps: int,
    zero_warps: int,
):
    del ZERO_BLOCK, scatter_warps, zero_warps  # unused in cuTile port
    (
        x0, index0, x1, index1, x2, index2, x3, index3,
        x4, index4, x5, index5, x6, index6, x7, index7,
        x8, index8, x9, index9, x10, index10, x11, index11,
        *_,
    ) = inputs

    device = x0.device
    outs = tuple(
        torch.zeros((BUCKETS, CHANNELS), device=device, dtype=torch.float32)
        for _ in range(12)
    )

    xs = (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)
    idxs = (index0, index1, index2, index3, index4, index5,
            index6, index7, index8, index9, index10, index11)
    for i in range(12):
        _launch_branch(xs[i], idxs[i], outs[i],
                       block_n=BLOCK_N, block_s=BLOCK_S)
    return outs
