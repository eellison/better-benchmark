"""cuTile port of sum_sum_sum_ed793bcaf7cf: DistillGPT2 LN-backward tail + dropout.

Structure matches Triton reference:
  Kernel 1 (_row_partial_kernel): for each M-row tile:
    - load x_bf16, addend, xhat, keep, scale, weight
    - compute x = addend + x_base, weighted = x * weight
    - row_sum = sum(weighted); row_dot = sum(weighted * xhat)
    - grad = scale * (weighted * D - row_sum - xhat * row_dot)
    - store grad_out (f32) and drop_out (bf16)
    - accumulate acc_x_xhat, acc_x, acc_drop into (num_tiles, D) partials
  Kernel 2 (_finalize_kernel): reduce (num_tiles, D) partials -> (D,)
    with the bf16-round of the drop sum at the end.

D=768 is not a power of 2, so we pad to BLOCK_D=1024 with zeros for reductions
and use padded scratch outputs that are narrowed back to D on return.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


D_LOGICAL = 768
D_PAD = 1024  # next pow2 >= 768


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@ct.kernel
def _row_partial_kernel(
    x_bf16_ptr,           # bf16 [M, D_PAD]
    addend_ptr,           # f32  [M, D_PAD]
    weight_ptr,           # f32  [D_PAD]
    xhat_ptr,             # f32  [M, D_PAD]
    scale_ptr,            # f32  [M]
    keep_ptr,             # bool [M, D_PAD] (as u8 in f32)
    grad_out_ptr,         # f32  [M, D_PAD]  (padded)
    drop_out_ptr,         # bf16 [M, D_PAD]  (padded)
    partial_x_xhat_ptr,   # f32  [NUM_TILES, D_PAD]
    partial_x_ptr,        # f32  [NUM_TILES, D_PAD]
    partial_drop_ptr,     # f32  [NUM_TILES, D_PAD]
    D_ACTUAL: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    tile_m = ct.bid(0)

    # Column mask: only cols < D_ACTUAL are valid.
    col_idx = ct.arange(BLOCK_D, dtype=ct.int32)
    col_mask_1d = col_idx < D_ACTUAL
    col_mask_2d = ct.reshape(col_mask_1d, (1, BLOCK_D))

    # Load weight [D_PAD] and mask off tail.
    weight_raw = ct.load(weight_ptr, index=(0,), shape=(BLOCK_D,))
    zero_1d = ct.zeros((BLOCK_D,), dtype=ct.float32)
    weight = ct.where(col_mask_1d, weight_raw, zero_1d)
    weight_2d = ct.reshape(weight, (1, BLOCK_D))

    # Load (BLOCK_M, BLOCK_D) tiles.
    x_base = ct.astype(
        ct.load(x_bf16_ptr, index=(tile_m, 0), shape=(BLOCK_M, BLOCK_D)),
        ct.float32,
    )
    addend = ct.load(addend_ptr, index=(tile_m, 0), shape=(BLOCK_M, BLOCK_D))
    xhat = ct.load(xhat_ptr, index=(tile_m, 0), shape=(BLOCK_M, BLOCK_D))
    keep = ct.astype(
        ct.load(keep_ptr, index=(tile_m, 0), shape=(BLOCK_M, BLOCK_D)),
        ct.float32,
    )
    scale = ct.load(scale_ptr, index=(tile_m,), shape=(BLOCK_M,))
    scale_2d = ct.reshape(scale, (BLOCK_M, 1))

    zero_2d = ct.zeros((BLOCK_M, BLOCK_D), dtype=ct.float32)
    x = addend + x_base
    weighted = x * weight_2d
    # Mask row reductions (axis=1 = D dim).
    weighted_masked = ct.where(col_mask_2d, weighted, zero_2d)
    row_sum = ct.sum(weighted_masked, axis=1, keepdims=True)  # (BLOCK_M, 1)
    weighted_xhat = weighted * xhat
    weighted_xhat_masked = ct.where(col_mask_2d, weighted_xhat, zero_2d)
    row_dot = ct.sum(weighted_xhat_masked, axis=1, keepdims=True)  # (BLOCK_M, 1)
    grad = scale_2d * (weighted * float(D_ACTUAL) - row_sum - xhat * row_dot)

    # Store to padded outputs (junk in tail is OK, torch narrows away).
    ct.store(grad_out_ptr, index=(tile_m, 0), tile=grad)
    dropped_bf = ct.astype(grad * keep * 1.1111111111111112, ct.bfloat16)
    ct.store(drop_out_ptr, index=(tile_m, 0), tile=dropped_bf)

    # Accumulators for column sums (reduce along axis=0 / M dim).
    # For the bf16-rounded drop-sum: match Triton's inner bf16 casts.
    grad_bf = ct.astype(grad, ct.bfloat16)
    keep_bf = ct.astype(keep, ct.bfloat16)
    ds_bf = ct.astype(
        ct.astype(keep_bf, ct.float32) * 1.1111111111111112,
        ct.bfloat16,
    )
    dropped_sum_bf = ct.astype(
        ct.astype(grad_bf, ct.float32) * ct.astype(ds_bf, ct.float32),
        ct.bfloat16,
    )
    dropped_sum_f = ct.astype(dropped_sum_bf, ct.float32)

    # Mask contributions in tail cols by zeroing.
    x_xhat_masked = ct.where(col_mask_2d, x * xhat, zero_2d)
    x_masked = ct.where(col_mask_2d, x, zero_2d)
    drop_masked = ct.where(col_mask_2d, dropped_sum_f, zero_2d)

    p_x_xhat = ct.sum(x_xhat_masked, axis=0)   # (BLOCK_D,)
    p_x = ct.sum(x_masked, axis=0)             # (BLOCK_D,)
    p_drop = ct.sum(drop_masked, axis=0)       # (BLOCK_D,)

    p_x_xhat_2d = ct.reshape(p_x_xhat, (1, BLOCK_D))
    p_x_2d = ct.reshape(p_x, (1, BLOCK_D))
    p_drop_2d = ct.reshape(p_drop, (1, BLOCK_D))
    ct.store(partial_x_xhat_ptr, index=(tile_m, 0), tile=p_x_xhat_2d)
    ct.store(partial_x_ptr, index=(tile_m, 0), tile=p_x_2d)
    ct.store(partial_drop_ptr, index=(tile_m, 0), tile=p_drop_2d)


@ct.kernel
def _finalize_kernel(
    partial_x_xhat_ptr,   # f32 [NUM_TILES, D_PAD]
    partial_x_ptr,        # f32 [NUM_TILES, D_PAD]
    partial_drop_ptr,     # f32 [NUM_TILES, D_PAD]
    out_x_xhat_ptr,       # f32 [D_PAD]
    out_x_ptr,            # f32 [D_PAD]
    out_drop_ptr,         # f32 [D_PAD]
    NUM_TILES: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
    NUM_TILES_PAD: ct.Constant[int],
):
    col_block = ct.bid(0)
    p_xx = ct.load(partial_x_xhat_ptr, index=(0, col_block),
                   shape=(NUM_TILES_PAD, BLOCK_D),
                   padding_mode=ct.PaddingMode.ZERO)
    p_x = ct.load(partial_x_ptr, index=(0, col_block),
                  shape=(NUM_TILES_PAD, BLOCK_D),
                  padding_mode=ct.PaddingMode.ZERO)
    p_d = ct.load(partial_drop_ptr, index=(0, col_block),
                  shape=(NUM_TILES_PAD, BLOCK_D),
                  padding_mode=ct.PaddingMode.ZERO)
    s_xx = ct.sum(p_xx, axis=0)
    s_x = ct.sum(p_x, axis=0)
    s_d = ct.sum(p_d, axis=0)
    s_d_bf = ct.astype(s_d, ct.bfloat16)
    s_d_out = ct.astype(s_d_bf, ct.float32)
    ct.store(out_x_xhat_ptr, index=(col_block,), tile=s_xx)
    ct.store(out_x_ptr, index=(col_block,), tile=s_x)
    ct.store(out_drop_ptr, index=(col_block,), tile=s_d_out)


def _ceil_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


# 9846b7f2: (T([16384,768], bf16), T([32,512,768], f32), T([768], f32), ...)
@oracle_impl(hardware="B200", point="9846b7f2", BLOCK_M=16, BLOCK_D=1024, FINAL_BLOCK_D=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_D: int,
    FINAL_BLOCK_D: int,
):
    x_bf16, addend, weight, xhat, scale, keep, shape0, shape1, shape2 = inputs
    b, s, d = _shape_tuple(shape0)
    m = b * s
    device = x_bf16.device

    # Requested outputs at logical D=768 shape.
    grad_out = torch.empty_strided(
        (b, s, d),
        (s * d, d, 1),
        device=device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        _shape_tuple(shape1),
        (d, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_x_xhat = torch.empty_strided(_shape_tuple(shape2), (1,), device=device, dtype=torch.float32)
    out_x = torch.empty_strided(_shape_tuple(shape2), (1,), device=device, dtype=torch.float32)
    out_drop = torch.empty_strided(_shape_tuple(shape2), (1,), device=device, dtype=torch.float32)

    if m % BLOCK_M != 0:
        raise NotImplementedError("BLOCK_M must divide M for cuTile port")
    num_tiles = m // BLOCK_M

    # Padded scratch tensors (BLOCK_D = D_PAD = 1024).
    D_PAD = BLOCK_D
    x_bf16_pad = torch.zeros((m, D_PAD), device=device, dtype=torch.bfloat16)
    x_bf16_pad[:, :d].copy_(x_bf16.view(m, d))
    addend_pad = torch.zeros((m, D_PAD), device=device, dtype=torch.float32)
    addend_pad[:, :d].copy_(addend.view(m, d))
    xhat_pad = torch.zeros((m, D_PAD), device=device, dtype=torch.float32)
    xhat_pad[:, :d].copy_(xhat.view(m, d) if xhat.shape != (m, d) else xhat)
    keep_pad = torch.zeros((m, D_PAD), device=device, dtype=keep.dtype)
    keep_pad[:, :d].copy_(keep.view(m, d) if keep.shape != (m, d) else keep)
    weight_pad = torch.zeros((D_PAD,), device=device, dtype=torch.float32)
    weight_pad[:d].copy_(weight)

    grad_out_pad = torch.empty((m, D_PAD), device=device, dtype=torch.float32)
    drop_out_pad = torch.empty((m, D_PAD), device=device, dtype=torch.bfloat16)

    partial_x_xhat = torch.empty((num_tiles, D_PAD), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_tiles, D_PAD), device=device, dtype=torch.float32)
    partial_drop = torch.empty((num_tiles, D_PAD), device=device, dtype=torch.float32)

    scale_1d = scale.view(m) if scale.numel() == m else scale

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_tiles, 1, 1),
        _row_partial_kernel,
        (
            x_bf16_pad, addend_pad, weight_pad, xhat_pad, scale_1d, keep_pad,
            grad_out_pad, drop_out_pad,
            partial_x_xhat, partial_x, partial_drop,
            d, BLOCK_M, BLOCK_D,
        ),
    )

    # Copy the valid D=768 columns from padded outputs back to the real outputs.
    grad_out.view(m, d).copy_(grad_out_pad[:, :d])
    drop_out.view(m, d).copy_(drop_out_pad[:, :d])

    # Finalize: pow-2 pad num_tiles for the load, then narrow output slot.
    num_tiles_pad = _ceil_pow2(num_tiles)
    out_x_xhat_pad = torch.empty((D_PAD,), device=device, dtype=torch.float32)
    out_x_pad = torch.empty((D_PAD,), device=device, dtype=torch.float32)
    out_drop_pad = torch.empty((D_PAD,), device=device, dtype=torch.float32)
    if D_PAD % FINAL_BLOCK_D != 0:
        raise NotImplementedError("FINAL_BLOCK_D must divide D_PAD")
    ct.launch(
        stream,
        (D_PAD // FINAL_BLOCK_D, 1, 1),
        _finalize_kernel,
        (
            partial_x_xhat, partial_x, partial_drop,
            out_x_xhat_pad, out_x_pad, out_drop_pad,
            num_tiles, FINAL_BLOCK_D, num_tiles_pad,
        ),
    )
    # Narrow finalized outputs.
    out_x_xhat.view(d).copy_(out_x_xhat_pad[:d])
    out_x.view(d).copy_(out_x_pad[:d])
    out_drop.view(d).copy_(out_drop_pad[:d])

    return grad_out, out_x_xhat, out_x, drop_out, out_drop
