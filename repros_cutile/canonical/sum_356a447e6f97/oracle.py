"""cuTile port of sum_356a447e6f97: PyTorch-UNet 2-channel spatial sum.

Input is bf16 `[1, 2, 640, 959]`, output is f32 `[2]` computed as the per-channel
sum with fp32 accumulation. We pad the spatial dim to a power of 2 and sum via
tiled cuTile reductions with intermediate partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW = 640 * 959  # 613760
BLOCK_ELEMENTS = 2048  # divisor-friendly block

# Number of tiles per channel — 613760 / 2048 = 299.6875, round up.
NUM_TILES = (HW + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS  # 300


@ct.kernel
def _pair_partial_sum_kernel(
    x_ptr,          # bf16 (2, PADDED_HW) - padded with zeros
    partial_ptr,    # f32 (2, NUM_TILES)
    BLOCK_ELEMENTS_C: ct.Constant[int],
):
    tile = ct.bid(0)
    ch0 = ct.load(x_ptr, index=(0, tile), shape=(1, BLOCK_ELEMENTS_C))
    ch1 = ct.load(x_ptr, index=(1, tile), shape=(1, BLOCK_ELEMENTS_C))
    ch0_f = ct.astype(ch0, ct.float32)
    ch1_f = ct.astype(ch1, ct.float32)
    s0 = ct.sum(ch0_f)
    s1 = ct.sum(ch1_f)
    s0t = ct.reshape(s0, (1, 1))
    s1t = ct.reshape(s1, (1, 1))
    ct.store(partial_ptr, index=(0, tile), tile=s0t)
    ct.store(partial_ptr, index=(1, tile), tile=s1t)


@ct.kernel
def _finalize_pair_sum_kernel(
    partial_ptr,    # f32 (2, NUM_TILES_PADDED)
    out_ptr,        # f32 (2,)
    NUM_TILES_C: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    ch0 = ct.load(partial_ptr, index=(0, 0), shape=(1, BLOCK_TILES))
    ch1 = ct.load(partial_ptr, index=(1, 0), shape=(1, BLOCK_TILES))
    t0 = ct.sum(ch0)
    t1 = ct.sum(ch1)
    t0t = ct.reshape(t0, (1,))
    t1t = ct.reshape(t1, (1,))
    ct.store(out_ptr, index=(0,), tile=t0t)
    ct.store(out_ptr, index=(1,), tile=t1t)
    # Note: NUM_TILES_C unused; kept for signature symmetry.


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="7a9f1993", BLOCK_ELEMENTS=BLOCK_ELEMENTS)
def oracle_forward(inputs, *, BLOCK_ELEMENTS: int):
    (arg0_1,) = inputs
    # Pad x to (2, NUM_TILES * BLOCK_ELEMENTS) with zeros in tail
    hw = int(arg0_1.shape[2]) * int(arg0_1.shape[3])
    padded_hw = NUM_TILES * BLOCK_ELEMENTS
    x_padded = torch.zeros((2, padded_hw), device=arg0_1.device, dtype=torch.bfloat16)
    x_padded[:, :hw] = arg0_1.view(2, hw)

    # Partials shape (2, NUM_TILES_POW2), padded with zeros
    tiles_pow2 = _next_pow2(NUM_TILES)
    partial = torch.zeros((2, tiles_pow2), device=arg0_1.device, dtype=torch.float32)

    out = torch.empty_strided((2,), (1,), device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUM_TILES, 1, 1),
        _pair_partial_sum_kernel,
        (x_padded, partial, BLOCK_ELEMENTS),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _finalize_pair_sum_kernel,
        (partial, out, NUM_TILES, tiles_pow2),
    )
    return out
