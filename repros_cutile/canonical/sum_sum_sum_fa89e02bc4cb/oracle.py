"""cuTile port of sum_sum_sum_fa89e02bc4cb: ConvNeXtV2 GRN backward.

Two cuTile kernels mirror Triton's structure:
  1. Producer kernel: per row (n, h, w) of NHWC data, computes row weighted-sum
     (sum_1 broadcast) and row weighted*grad sum (sum_2 broadcast) via ct.sum
     over channels; writes NHWC->NCHW-stride f32 output (mul_4), the bf16 view,
     and per-column partial sums (x*grad, x, y_bf16.f32) for the three
     channel-reductions.
  2. Finalize kernel: reduces the per-tile column partials -> [C] results
     (sum_3, sum_4, and bf16-rounded sum_5 -> f32).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C_ = 80
BLOCK_C = 128  # next pow-of-2 of 80
N_CONST = 80.0


def _ceil_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


@ct.kernel
def _grn_producer_kernel(
    x_ptr,               # bf16 [rows, C] NHWC contig
    weight_ptr,          # f32 [C]
    grad_ptr,            # f32 [rows, C] NHWC contig (from arg2_1)
    scale_ptr,           # f32 [rows] arg3_1 flat
    y_f32_ptr,           # f32 [rows, C] NHWC contig
    y_bf16_ptr,          # bf16 [rows, C] NHWC contig
    partials_ptr,        # f32 [3, NUM_ROW_TILES, C_PAD] flat -> (partial, tile, c)
    ROWS_: ct.Constant[int],
    C_CT: ct.Constant[int],
    NUM_ROW_TILES_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C_CT: ct.Constant[int],
    N_CONST_: ct.Constant[float],
):
    tile = ct.bid(0)
    row_start = tile * BLOCK_R
    row_ids = row_start + ct.arange(BLOCK_R, dtype=ct.int32)
    row_lim = ct.full(shape=(BLOCK_R,), fill_value=ROWS_, dtype=ct.int32)
    row_valid = row_ids < row_lim

    col_ids = ct.arange(BLOCK_C_CT, dtype=ct.int32)
    col_lim = ct.full(shape=(BLOCK_C_CT,), fill_value=C_CT, dtype=ct.int32)
    col_valid = col_ids < col_lim

    row_valid_2d = ct.reshape(row_valid, (BLOCK_R, 1))
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C_CT))
    tile_valid = row_valid_2d & col_valid_2d

    x_bf = ct.load(
        x_ptr, index=(tile, 0), shape=(BLOCK_R, BLOCK_C_CT),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x_bf, ct.float32)
    grad = ct.load(
        grad_ptr, index=(tile, 0), shape=(BLOCK_R, BLOCK_C_CT),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_C_CT,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_C_CT))
    scale = ct.load(
        scale_ptr, index=(tile,), shape=(BLOCK_R,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scale_2d = ct.reshape(scale, (BLOCK_R, 1))

    weighted = x_f * weight_2d
    weighted_grad = weighted * grad
    zeros_2d = ct.zeros((BLOCK_R, BLOCK_C_CT), dtype=ct.float32)
    weighted_m = ct.where(tile_valid, weighted, zeros_2d)
    weighted_grad_m = ct.where(tile_valid, weighted_grad, zeros_2d)
    row_weighted_sum = ct.sum(weighted_m, axis=1, keepdims=True)     # (BLOCK_R, 1)
    row_weighted_grad_sum = ct.sum(weighted_grad_m, axis=1, keepdims=True)

    lhs = weighted * N_CONST_ - row_weighted_sum
    rhs = grad * row_weighted_grad_sum
    value = scale_2d * (lhs - rhs)
    value_bf = ct.astype(value, ct.bfloat16)
    ct.store(y_f32_ptr, index=(tile, 0), tile=value)
    ct.store(y_bf16_ptr, index=(tile, 0), tile=value_bf)

    # Per-column partial sums: (x*grad), x, and value_bf.f32 for sum_5.
    xg = x_f * grad
    xg_m = ct.where(tile_valid, xg, zeros_2d)
    x_m = ct.where(tile_valid, x_f, zeros_2d)
    y_bf_f = ct.astype(value_bf, ct.float32)
    y_bf_f_m = ct.where(tile_valid, y_bf_f, zeros_2d)

    sum_xg = ct.sum(xg_m, axis=0, keepdims=False)         # (BLOCK_C_CT,)
    sum_x = ct.sum(x_m, axis=0, keepdims=False)
    sum_yf = ct.sum(y_bf_f_m, axis=0, keepdims=False)

    tile_off = tile * BLOCK_C_CT
    partials_stride = NUM_ROW_TILES_ * BLOCK_C_CT
    base = ct.full(shape=(BLOCK_C_CT,), fill_value=tile_off, dtype=ct.int32)
    off_xg = base + col_ids
    off_x = base + col_ids + partials_stride
    off_y = base + col_ids + 2 * partials_stride
    ct.scatter(partials_ptr, off_xg, sum_xg, mask=col_valid)
    ct.scatter(partials_ptr, off_x, sum_x, mask=col_valid)
    ct.scatter(partials_ptr, off_y, sum_yf, mask=col_valid)


@ct.kernel
def _grn_finalize_kernel(
    part_xg_ptr,         # f32 [NUM_ROW_TILES, BLOCK_C_CT]
    part_x_ptr,          # f32 [NUM_ROW_TILES, BLOCK_C_CT]
    part_y_ptr,          # f32 [NUM_ROW_TILES, BLOCK_C_CT]
    out_sum_xgrad_ptr,   # f32 [C]
    out_sum_x_ptr,       # f32 [C]
    out_sum_y_ptr,       # f32 [C]
    C_CT: ct.Constant[int],
    NUM_ROW_TILES_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C_CT: ct.Constant[int],
):
    c_block = ct.bid(0)
    c_ids = c_block * BLOCK_C_CT + ct.arange(BLOCK_C_CT, dtype=ct.int32)
    c_lim = ct.full(shape=(BLOCK_C_CT,), fill_value=C_CT, dtype=ct.int32)
    c_valid = c_ids < c_lim

    tiles_1d = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_lim = ct.full(shape=(BLOCK_TILES,), fill_value=NUM_ROW_TILES_, dtype=ct.int32)
    tile_valid_1d = tiles_1d < tile_lim
    tile_valid_2d = ct.reshape(tile_valid_1d, (BLOCK_TILES, 1))
    c_valid_2d = ct.reshape(c_valid, (1, BLOCK_C_CT))
    valid = tile_valid_2d & c_valid_2d

    part_xg = ct.load(
        part_xg_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C_CT),
        padding_mode=ct.PaddingMode.ZERO,
    )
    part_x = ct.load(
        part_x_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C_CT),
        padding_mode=ct.PaddingMode.ZERO,
    )
    part_y = ct.load(
        part_y_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C_CT),
        padding_mode=ct.PaddingMode.ZERO,
    )

    zeros = ct.zeros((BLOCK_TILES, BLOCK_C_CT), dtype=ct.float32)
    xg_m = ct.where(valid, part_xg, zeros)
    x_m = ct.where(valid, part_x, zeros)
    y_m = ct.where(valid, part_y, zeros)
    sum_xg = ct.sum(xg_m, axis=0, keepdims=False)     # (BLOCK_C_CT,)
    sum_x = ct.sum(x_m, axis=0, keepdims=False)
    sum_y = ct.sum(y_m, axis=0, keepdims=False)

    # bf16 rounding for sum_5.
    sum_y_bf = ct.astype(sum_y, ct.bfloat16)
    sum_y_final = ct.astype(sum_y_bf, ct.float32)

    c_offs = c_block * BLOCK_C_CT + ct.arange(BLOCK_C_CT, dtype=ct.int32)
    ct.scatter(out_sum_xgrad_ptr, c_offs, sum_xg, mask=c_valid)
    ct.scatter(out_sum_x_ptr, c_offs, sum_x, mask=c_valid)
    ct.scatter(out_sum_y_ptr, c_offs, sum_y_final, mask=c_valid)


@oracle_impl(hardware="B200", point="535103c6", BLOCK_R=128)
def oracle_forward(inputs, *, BLOCK_R: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    device = arg0_1.device
    n, c, h, w = int(arg0_1.shape[0]), int(arg0_1.shape[1]), int(arg0_1.shape[2]), int(arg0_1.shape[3])
    rows = n * h * w
    assert c == C_

    # NHWC view of arg0_1 (channels-last input). This is contig in that layout.
    x_nhwc_2d = arg0_1.permute(0, 2, 3, 1).contiguous().view(rows, c)
    # arg2_1 is [n, h, w, c] with special strides — permute so channels are last-contig.
    # Actual memory layout: (batch, w, c, h) — permute (0, 2, 3, 1) yields (b, w, h, c).
    # But shape order is (n, h, w, c). We want (rows, c) where rows = n*h*w.
    # Simplest: .contiguous() and view.
    grad_nhwc_2d = arg2_1.contiguous().view(rows, c)
    scale_flat = arg3_1.contiguous().view(rows)

    # Output y is NCHW-strided f32 (matches the eager mul_4 output which is permuted
    # back via .permute(0, 3, 1, 2)).
    # For the kernel, produce NHWC-contiguous f32 [rows, c], then permute to NCHW.
    y_nhwc = torch.empty((rows, c), device=device, dtype=torch.float32)
    y_bf16_nhwc = torch.empty((rows, c), device=device, dtype=torch.bfloat16)

    num_row_tiles = (rows + BLOCK_R - 1) // BLOCK_R
    part_xg = torch.empty((num_row_tiles, BLOCK_C), device=device, dtype=torch.float32)
    part_x = torch.empty((num_row_tiles, BLOCK_C), device=device, dtype=torch.float32)
    part_y = torch.empty((num_row_tiles, BLOCK_C), device=device, dtype=torch.float32)
    partials_flat = torch.empty((3 * num_row_tiles * BLOCK_C,), device=device, dtype=torch.float32)
    # Reuse a single storage: bind three slices via view.
    part_xg = partials_flat[:num_row_tiles * BLOCK_C].view(num_row_tiles, BLOCK_C)
    part_x = partials_flat[num_row_tiles * BLOCK_C:2 * num_row_tiles * BLOCK_C].view(num_row_tiles, BLOCK_C)
    part_y = partials_flat[2 * num_row_tiles * BLOCK_C:].view(num_row_tiles, BLOCK_C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_tiles, 1, 1), _grn_producer_kernel,
        (x_nhwc_2d, arg1_1, grad_nhwc_2d, scale_flat,
         y_nhwc, y_bf16_nhwc, partials_flat,
         rows, c, num_row_tiles, BLOCK_R, BLOCK_C, N_CONST),
    )

    sum_3 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((c,), device=device, dtype=torch.float32)
    sum_5_f32 = torch.empty((c,), device=device, dtype=torch.float32)
    block_tiles = _ceil_pow2(num_row_tiles)
    ct.launch(
        stream, ((c + BLOCK_C - 1) // BLOCK_C, 1, 1), _grn_finalize_kernel,
        (part_xg, part_x, part_y, sum_3, sum_4, sum_5_f32,
         c, num_row_tiles, block_tiles, BLOCK_C),
    )

    # Rearrange y_nhwc [rows, c] -> [n, c, h, w] with NCHW strides preserved.
    # First reshape to (n, h, w, c), then permute to (n, c, h, w).
    y_nhwc_4d = y_nhwc.view(n, h, w, c).permute(0, 3, 1, 2)  # (n, c, h, w) with strided view
    permute_1 = y_nhwc_4d.contiguous()
    convert_element_type_1 = y_bf16_nhwc.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    return sum_3, sum_4, permute_1, convert_element_type_1, sum_5_f32
