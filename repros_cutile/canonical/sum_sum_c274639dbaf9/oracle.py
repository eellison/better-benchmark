"""cuTile port of sum_sum_c274639dbaf9: SqueezeNet dual slice/where + channel sum.

Two-kernel design mirrors Triton: kernel 1 does dual where + partial channel sum,
kernel 2 finalizes channel sums. Rows = N*H*W = 32*27*27 = 23328 is not a power
of 2; largest pow2 divisor is 32, which we use as BLOCK_R.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _dual_where_partial_kernel(
    slice_left_ptr,     # bf16 (rows, C) contiguous
    slice_right_ptr,    # bf16 (rows, C) contiguous
    mask_left_ptr,      # bool (rows, C) contiguous
    mask_right_ptr,     # bool (rows, C) contiguous
    fill_ptr,           # bf16 (1,)
    out_left_ptr,       # bf16 (rows, C) contiguous
    out_right_ptr,      # bf16 (rows, C) contiguous
    partial_left_ptr,   # f32 (row_tiles, C)
    partial_right_ptr,  # f32 (row_tiles, C)
    ROWS: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    c_block = ct.bid(1)

    slice_left = ct.load(slice_left_ptr, index=(row_block, c_block), shape=(BLOCK_R, BLOCK_C))
    slice_right = ct.load(slice_right_ptr, index=(row_block, c_block), shape=(BLOCK_R, BLOCK_C))
    mask_left = ct.load(mask_left_ptr, index=(row_block, c_block), shape=(BLOCK_R, BLOCK_C))
    mask_right = ct.load(mask_right_ptr, index=(row_block, c_block), shape=(BLOCK_R, BLOCK_C))
    fill_v = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_b = ct.reshape(fill_v, (1, 1))

    left_out = ct.where(mask_left, fill_b, slice_left)
    right_out = ct.where(mask_right, fill_b, slice_right)

    ct.store(out_left_ptr, index=(row_block, c_block), tile=left_out)
    ct.store(out_right_ptr, index=(row_block, c_block), tile=right_out)

    left_f = ct.astype(left_out, ct.float32)
    right_f = ct.astype(right_out, ct.float32)
    left_sum = ct.sum(left_f, axis=0)   # (BLOCK_C,)
    right_sum = ct.sum(right_f, axis=0)

    ct.store(partial_left_ptr, index=(row_block, c_block),
             tile=ct.reshape(left_sum, (1, BLOCK_C)))
    ct.store(partial_right_ptr, index=(row_block, c_block),
             tile=ct.reshape(right_sum, (1, BLOCK_C)))


@ct.kernel
def _dual_where_final_kernel(
    partial_left_ptr,      # f32 (row_tiles, C)
    partial_right_ptr,     # f32 (row_tiles, C)
    out_left_sum_ptr,      # f32 (C,)
    out_right_sum_ptr,     # f32 (C,)
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    left_tile = ct.load(
        partial_left_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    right_tile = ct.load(
        partial_right_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    left_total = ct.sum(left_tile, axis=0)
    right_total = ct.sum(right_tile, axis=0)
    # Round through bf16 to match Repro's bf16-accumulated sum boundary.
    left_rounded = ct.astype(ct.astype(left_total, ct.bfloat16), ct.float32)
    right_rounded = ct.astype(ct.astype(right_total, ct.bfloat16), ct.float32)
    ct.store(out_left_sum_ptr, index=(c_block,), tile=left_rounded)
    ct.store(out_right_sum_ptr, index=(c_block,), tile=right_rounded)


def _next_power_of_2(v):
    return 1 << (int(v) - 1).bit_length()


# 400df623: torchbench squeezenet1_1 train, bf16 [32, 256, 27, 27] channels-last input.
@oracle_impl(hardware="B200", point="400df623", BLOCK_R=32, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    arg0, mask_right, fill, mask_left = inputs

    n = 32
    c = 128
    h = 27
    w = 27
    rows = n * h * w  # 23328

    # Materialize NHWC-contiguous versions of the slice halves and masks.
    arg0_nhwc = arg0.permute(0, 2, 3, 1).contiguous()  # [32, 27, 27, 256]
    slice_left_nhwc = arg0_nhwc[..., 0:128].contiguous()
    slice_right_nhwc = arg0_nhwc[..., 128:256].contiguous()
    mask_right_nhwc = mask_right.permute(0, 2, 3, 1).contiguous()
    mask_left_nhwc = mask_left.permute(0, 2, 3, 1).contiguous()

    slice_left_2d = slice_left_nhwc.view(rows, c)
    slice_right_2d = slice_right_nhwc.view(rows, c)
    mask_left_2d = mask_left_nhwc.view(rows, c)
    mask_right_2d = mask_right_nhwc.view(rows, c)

    def _largest_pow2_divisor(nn, cap):
        d = 1
        while d * 2 <= cap and nn % (d * 2) == 0:
            d *= 2
        return d
    block_r = _largest_pow2_divisor(rows, BLOCK_R)
    block_c = _largest_pow2_divisor(c, BLOCK_C)
    row_tiles = rows // block_r  # 729
    c_tiles = c // block_c       # 8
    block_tiles = _next_power_of_2(row_tiles)  # 1024

    # Allocate NHWC-contiguous output storage; view for kernel and permute back.
    out_left_nhwc = torch.empty((n, h, w, c), device=arg0.device, dtype=torch.bfloat16)
    out_right_nhwc = torch.empty((n, h, w, c), device=arg0.device, dtype=torch.bfloat16)
    out_left_2d = out_left_nhwc.view(rows, c)
    out_right_2d = out_right_nhwc.view(rows, c)

    partial_left = torch.empty((row_tiles, c), device=arg0.device, dtype=torch.float32)
    partial_right = torch.empty((row_tiles, c), device=arg0.device, dtype=torch.float32)

    out_left_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out_right_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)

    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (row_tiles, c_tiles, 1),
        _dual_where_partial_kernel,
        (slice_left_2d, slice_right_2d, mask_left_2d, mask_right_2d, fill_1d,
         out_left_2d, out_right_2d, partial_left, partial_right,
         rows, c, block_r, block_c),
    )
    ct.launch(
        stream,
        (c_tiles, 1, 1),
        _dual_where_final_kernel,
        (partial_left, partial_right, out_left_sum, out_right_sum,
         block_tiles, block_c),
    )

    # Channels-last strided view (no copy since storage is NHWC-contiguous).
    out_left = out_left_nhwc.permute(0, 3, 1, 2)
    out_right = out_right_nhwc.permute(0, 3, 1, 2)

    return out_right, out_right_sum, out_left, out_left_sum
