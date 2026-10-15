"""cuTile port of sum_sum_sum_865e7fa30b8a: ConvNeXtV2 GRN backward multi-reduction.

Structure:
  1. Producer kernel: for each row (pixel) chunk, compute per-row sums and the
     bf16 gradient tile `y`. Emit three per-channel partial sums per row-tile
     (sum_x*grad_scaled, sum_x, sum_y_bf16).
  2. Finalizer kernel: reduce the per-row-tile partials along tile axis to
     get final channel sums (sum_3, sum_4, sum_5).

Simplifications vs the Triton oracle:
  * The channels-last inputs are torch-permuted to contiguous NHWC first;
    strides then become natural.
  * The tile-space uses (ROW_TILES, C_PADDED); C is padded to next pow2 and
    masked for reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


@ct.kernel
def _producer_kernel(
    x_nhwc,        # bf16 (ROWS, C_PADDED)  (rows = N*H*W)
    grad_nhwc,     # bf16 (ROWS, C_PADDED)
    weight,        # f32  (C_PADDED,)
    mean,          # f32  (ROWS,)
    scale,         # f32  (ROWS,)
    y_nhwc,        # bf16 (ROWS, C_PADDED)
    partial_xgrad, # f32  (NUM_TILES, C_PADDED)
    partial_x,     # f32  (NUM_TILES, C_PADDED)
    partial_y,     # f32  (NUM_TILES, C_PADDED)
    channels: ct.Constant[int],
    channels_f: ct.Constant[float],
    inv_channels: ct.Constant[float],
    ROWS: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)

    # Load a (BLOCK_R, BLOCK_C) tile at (tile*BLOCK_R : tile*BLOCK_R + BLOCK_R, 0 : BLOCK_C).
    x = ct.load(
        x_nhwc, index=(tile, 0), shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    grad = ct.load(
        grad_nhwc, index=(tile, 0), shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    grad_f = ct.astype(grad, ct.float32)

    # Per-row scalars. Index in tile-space: `tile` is the partition ID.
    m = ct.load(mean, index=(tile,), shape=(BLOCK_R,),
                padding_mode=ct.PaddingMode.ZERO)
    s = ct.load(scale, index=(tile,), shape=(BLOCK_R,),
                padding_mode=ct.PaddingMode.ZERO)
    m_col = ct.reshape(m, (BLOCK_R, 1))
    s_col = ct.reshape(s, (BLOCK_R, 1))

    w = ct.load(weight, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO)
    w_row = ct.reshape(w, (1, BLOCK_C))

    # Masks.
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = c_idx < channels
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_C))
    # Rows are already all-in-bounds for the shapes we support (rows % BLOCK_R
    # == 0). So the active mask is just the column mask broadcast to rows.
    active = col_mask_2d

    centered = grad_f - m_col
    grad_scaled = centered * s_col
    weighted = x_f * w_row
    weighted_grad = weighted * grad_scaled

    # Per-row reductions (axis=1, channel axis).
    row_weighted_sum = ct.sum(ct.where(active, weighted, 0.0), axis=1, keepdims=True)
    row_weighted_grad_sum = ct.sum(
        ct.where(active, weighted_grad, 0.0), axis=1, keepdims=True
    )

    lhs = weighted * channels_f - row_weighted_sum
    rhs = grad_scaled * row_weighted_grad_sum
    value = (s_col * inv_channels) * (lhs - rhs)
    value_bf = ct.astype(value, ct.bfloat16)
    ct.store(y_nhwc, index=(tile, 0), tile=value_bf)

    # Per-column partial reductions (axis=0, row axis).
    xgrad_masked = ct.where(active, x_f * grad_scaled, 0.0)
    x_masked = ct.where(active, x_f, 0.0)
    y_f = ct.astype(value_bf, ct.float32)
    y_masked = ct.where(active, y_f, 0.0)

    p_xgrad = ct.sum(xgrad_masked, axis=0, keepdims=True)
    p_x = ct.sum(x_masked, axis=0, keepdims=True)
    p_y = ct.sum(y_masked, axis=0, keepdims=True)

    ct.store(partial_xgrad, index=(tile, 0), tile=p_xgrad)
    ct.store(partial_x, index=(tile, 0), tile=p_x)
    ct.store(partial_y, index=(tile, 0), tile=p_y)


@ct.kernel
def _finalize_kernel(
    partial_xgrad,  # f32 (NUM_TILES, C_PADDED)
    partial_x,
    partial_y,
    out_xgrad,      # f32 (C_PADDED,)
    out_x,
    out_y,
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    px = ct.load(
        partial_xgrad, index=(0, col_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    py = ct.load(
        partial_x, index=(0, col_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pz = ct.load(
        partial_y, index=(0, col_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    t_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    t_mask = t_idx < NUM_TILES
    t_mask_2d = ct.reshape(t_mask, (BLOCK_TILES, 1))
    sum_xgrad = ct.sum(ct.where(t_mask_2d, px, 0.0), axis=0)
    sum_x = ct.sum(ct.where(t_mask_2d, py, 0.0), axis=0)
    sum_y_f = ct.sum(ct.where(t_mask_2d, pz, 0.0), axis=0)
    # sum_y is finalized via bf16 round-trip in Triton
    sum_y_bf = ct.astype(sum_y_f, ct.bfloat16)
    sum_y_rr = ct.astype(sum_y_bf, ct.float32)

    ct.store(out_xgrad, index=(col_block,), tile=sum_xgrad)
    ct.store(out_x, index=(col_block,), tile=sum_x)
    ct.store(out_y, index=(col_block,), tile=sum_y_rr)


@oracle_impl(hardware="B200", point="5cee1fdd", BLOCK_R=128, BLOCK_C=128, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="9deba5e9", BLOCK_R=32, BLOCK_C=256, FINAL_BLOCK_C=2)
@oracle_impl(hardware="B200", point="2cbe1bb4", BLOCK_R=16, BLOCK_C=512, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="d5a3644b", BLOCK_R=16, BLOCK_C=1024, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_R, FINAL_BLOCK_C):
    arg0, arg1, arg2, arg3, arg4 = inputs
    n = int(arg0.shape[0])
    c = int(arg0.shape[1])
    h = int(arg0.shape[2])
    w = int(arg0.shape[3])
    rows = n * h * w
    c_padded = _next_pow2(c)
    if c_padded < BLOCK_C:
        c_padded = BLOCK_C
    num_tiles = (rows + BLOCK_R - 1) // BLOCK_R

    # Bring channels-last inputs into contiguous (rows, C) view.
    x_nhwc = arg0.permute(0, 2, 3, 1).contiguous().view(rows, c)
    grad_nhwc = arg2.permute(0, 2, 3, 1).contiguous().view(rows, c)
    # mean/scale have shape (N, H, W, 1) → flatten to (rows,).
    mean_flat = arg3.contiguous().view(rows)
    scale_flat = arg4.contiguous().view(rows)

    def _pad_c(t):
        if t.shape[-1] == c_padded:
            return t
        padded = torch.zeros(
            (*t.shape[:-1], c_padded), device=t.device, dtype=t.dtype
        )
        padded[..., :c] = t
        return padded

    x_padded = _pad_c(x_nhwc)
    grad_padded = _pad_c(grad_nhwc)
    weight_padded = _pad_c(arg1.contiguous())

    y_padded = torch.empty(
        (rows, c_padded), device=arg0.device, dtype=torch.bfloat16
    )
    partial_xgrad = torch.zeros(
        (num_tiles, c_padded), device=arg0.device, dtype=torch.float32
    )
    partial_x = torch.zeros(
        (num_tiles, c_padded), device=arg0.device, dtype=torch.float32
    )
    partial_y = torch.zeros(
        (num_tiles, c_padded), device=arg0.device, dtype=torch.float32
    )
    sum_xgrad = torch.empty((c_padded,), device=arg0.device, dtype=torch.float32)
    sum_x = torch.empty((c_padded,), device=arg0.device, dtype=torch.float32)
    sum_y = torch.empty((c_padded,), device=arg0.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_tiles, 1, 1), _producer_kernel,
        (x_padded, grad_padded, weight_padded, mean_flat, scale_flat,
         y_padded, partial_xgrad, partial_x, partial_y,
         c, 80.0, 1.0 / 80.0, rows, BLOCK_R, c_padded),
    )
    ct.launch(
        stream, ((c_padded + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_kernel,
        (partial_xgrad, partial_x, partial_y,
         sum_xgrad, sum_x, sum_y,
         num_tiles, _next_pow2(num_tiles), FINAL_BLOCK_C),
    )

    # Slice off channel padding.
    sum_xgrad_final = sum_xgrad[:c].contiguous()
    sum_x_final = sum_x[:c].contiguous()
    sum_y_final = sum_y[:c].contiguous()

    # y is (rows, C_PADDED). Take first C cols, view (N, H, W, C), permute
    # to (N, C, H, W) with the exact eager stride.
    y_hwc = y_padded[:, :c].contiguous().view(n, h, w, c)
    # The returned tensor has strides
    # (arg0.stride(0), arg0.stride(1), arg0.stride(3), arg0.stride(2)) —
    # i.e., NHWC-shaped with same strides as arg0 permuted. Just build it as
    # channels-last: (N, C, H, W) with stride (C*H*W, 1, W*C, C).
    y_out = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, w * c, c),
        device=arg0.device, dtype=torch.bfloat16,
    )
    y_out.copy_(y_hwc.permute(0, 3, 1, 2))
    return sum_xgrad_final, sum_x_final, y_out, sum_y_final
