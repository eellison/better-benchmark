"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete PyTorch-UNet bf16 `[1,2,640,959] -> [2]` channel-sum scope with a shape-specialized split reduction, reading each large contiguous channel slab once into fp32 partials and finalizing the two returned f32 sums in a second tiny kernel, whereas Inductor lowers this isolated reduction through its generic reduction template for a very small output and large spatial extent; Inductor cannot do this today because its scheduler does not select a dedicated two-channel split-K plan for this fixed layout and reduction extent; the fix is COOPERATIVE_SPLIT_K: add a guarded small-channel large-spatial reduction schedule that computes per-channel partials cooperatively and finalizes the tiny output vector directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _pair_partial_sum_kernel(
    x_ptr,
    partial_ptr,
    HW: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)
    mask = offsets < HW

    channel0 = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    channel1 = tl.load(x_ptr + HW + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + tile, tl.sum(channel0, axis=0))
    tl.store(partial_ptr + NUM_TILES + tile, tl.sum(channel1, axis=0))


@triton.jit
def _finalize_pair_sum_kernel(
    partial_ptr,
    out_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    channel0 = tl.load(partial_ptr + tiles, mask=mask, other=0.0).to(tl.float32)
    channel1 = tl.load(partial_ptr + NUM_TILES + tiles, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(channel0, axis=0))
    tl.store(out_ptr + 1, tl.sum(channel1, axis=0))


@oracle_impl(
    hardware="B200",
    point="7a9f1993",
    BLOCK_ELEMENTS=2048,
    partial_warps=2,
    final_warps=1,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_ELEMENTS: int,
    partial_warps: int,
    final_warps: int,
):
    (arg0_1,) = inputs
    hw = int(arg0_1.shape[2]) * int(arg0_1.shape[3])
    num_tiles = triton.cdiv(hw, BLOCK_ELEMENTS)

    partial = torch.empty_strided(
        (2, num_tiles),
        (num_tiles, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((2,), (1,), device=arg0_1.device, dtype=torch.float32)

    _pair_partial_sum_kernel[(num_tiles,)](
        arg0_1,
        partial,
        HW=hw,
        NUM_TILES=num_tiles,
        BLOCK_ELEMENTS=BLOCK_ELEMENTS,
        num_warps=partial_warps,
        num_stages=3,
    )
    _finalize_pair_sum_kernel[(1,)](
        partial,
        out,
        NUM_TILES=num_tiles,
        BLOCK_TILES=_ceil_pow2(num_tiles),
        num_warps=final_warps,
        num_stages=3,
    )
    return out
