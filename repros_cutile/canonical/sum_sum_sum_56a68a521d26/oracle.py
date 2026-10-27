"""cuTile port of sum_sum_sum_56a68a521d26: ConvBERT LN-backward + dropout.

Split into two cuTile kernels:
  Kernel A: per-row LN-backward + dropout side output. One program per
    row (16384 rows). Reads 6 inputs (one is transposed), reduces across
    columns, computes grad, writes grad_out and bf16 side, accumulates
    per-row contributions to a per-row-partial [rows, 3, channels]? No —
    simpler: use atomic_add against per-channel accumulators.
  Kernel B: reduce accumulator arrays into the output vectors.

Actually simpler than the Triton oracle since cuTile has atomic_add:
  Kernel: per-row. For each row:
    - build 'x' by adding 6 producers (5 direct + one strided)
    - compute weighted = x * weight
    - row_sum = sum(weighted); row_dot = sum(weighted * rhs)
    - centered = weighted * 768 - row_sum - rhs * row_dot
    - grad = scale * centered
    - side = grad * (keep * DROP_SCALE)  (bf16 rounded twice)
    - store grad, store side
    - atomic_add per-column accumulators (out_x_rhs += x*rhs, out_x += x,
      out_bf16_sum += side_bf16)
  Final: for out_bf16_sum, do one extra bf16 round trip.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
CHANNELS = 768
ROW_FACTOR = 768.0
DROP_SCALE = 1.1111111111111112


@ct.kernel
def _ln_backward_row_kernel(
    arg0_ptr,        # bf16 [rows, BLOCK_C] padded
    arg1_ptr,        # f32  [rows, BLOCK_C] padded
    arg2_ptr,        # bf16 [rows, BLOCK_C] padded
    arg3_perm_ptr,   # bf16 [rows, BLOCK_C] padded
    arg4_ptr,        # bf16 [rows, BLOCK_C] padded
    arg5_ptr,        # bf16 [rows, BLOCK_C] padded
    weight_ptr,      # f32  [BLOCK_C] padded
    rhs_ptr,         # f32  [rows, BLOCK_C] padded
    scale_ptr,       # f32  [rows]
    mask_ptr,        # b8   [rows, BLOCK_C] padded
    grad_out_ptr,    # f32  [rows, BLOCK_C] padded
    bf16_out_ptr,    # bf16 [rows, BLOCK_C] padded
    out_x_rhs_ptr,   # f32  [channels] (atomic accumulator)
    out_x_ptr,       # f32  [channels] (atomic accumulator)
    out_bf16_sum_ptr,# f32  [channels] (atomic accumulator)
    CHANNELS_: ct.Constant[int],
    ROW_FACTOR_: ct.Constant[float],
    DROP_SCALE_: ct.Constant[float],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)

    # Load full row via padded tile.
    a0 = ct.astype(ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_C)), ct.float32)
    a1 = ct.load(arg1_ptr, index=(row, 0), shape=(1, BLOCK_C))
    a2 = ct.astype(ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_C)), ct.float32)
    a3 = ct.astype(ct.load(arg3_perm_ptr, index=(row, 0), shape=(1, BLOCK_C)), ct.float32)
    a4 = ct.astype(ct.load(arg4_ptr, index=(row, 0), shape=(1, BLOCK_C)), ct.float32)
    a5 = ct.astype(ct.load(arg5_ptr, index=(row, 0), shape=(1, BLOCK_C)), ct.float32)
    x = a1 + a0 + a2 + a3 + a4 + a5

    # Column mask
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_valid = cols < CHANNELS_
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C))

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,)), ct.float32)
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_C))
    scale = ct.load(scale_ptr, index=(row,), shape=(1,))
    keep = ct.astype(ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_C)), ct.float32)

    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    weighted = x * weight_2d
    weighted_masked = ct.where(col_valid_2d, weighted, 0.0)
    row_sum = ct.sum(weighted_masked, axis=1, keepdims=True)
    weighted_rhs = weighted * rhs
    weighted_rhs_masked = ct.where(col_valid_2d, weighted_rhs, 0.0)
    row_dot = ct.sum(weighted_rhs_masked, axis=1, keepdims=True)
    centered = weighted * ROW_FACTOR_ - row_sum - rhs * row_dot
    scale_2d = ct.reshape(scale, (1, 1))
    grad = scale_2d * centered
    grad_masked = ct.where(col_valid_2d, grad, 0.0)
    ct.store(grad_out_ptr, index=(row, 0), tile=grad_masked)

    grad_bf = ct.astype(grad, ct.bfloat16)
    keep_scale = ct.astype(keep * DROP_SCALE_, ct.bfloat16)
    side_bf = ct.astype(
        ct.astype(grad_bf, ct.float32) * ct.astype(keep_scale, ct.float32),
        ct.bfloat16,
    )
    store_side_bf = ct.astype(
        grad * ct.astype(keep_scale, ct.float32),
        ct.bfloat16,
    )
    ct.store(bf16_out_ptr, index=(row, 0), tile=store_side_bf)

    # Per-column accumulators via atomic_add. Only valid columns.
    x_flat = ct.reshape(x, (BLOCK_C,))
    rhs_flat = ct.reshape(rhs, (BLOCK_C,))
    side_flat = ct.reshape(ct.astype(side_bf, ct.float32), (BLOCK_C,))
    # Use out-of-bounds column index for invalid columns → atomic dropped.
    invalid_cols = ct.full((BLOCK_C,), CHANNELS_, dtype=ct.int32)
    cols_safe = ct.where(col_valid, cols, invalid_cols)
    ct.atomic_add(out_x_rhs_ptr, (cols_safe,), x_flat * rhs_flat)
    ct.atomic_add(out_x_ptr, (cols_safe,), x_flat)
    ct.atomic_add(out_bf16_sum_ptr, (cols_safe,), side_flat)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="730e8332", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=2)
def oracle_forward(inputs, *, ROWS_PER_GROUP, BLOCK_R, BLOCK_C, FINAL_BLOCK_C):
    del ROWS_PER_GROUP, BLOCK_R, FINAL_BLOCK_C
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        arg6_1, arg7_1, arg8_1, arg9_1,
        _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3,
        shape_flat, shape_sum_bf16,
    ) = inputs

    full_shape = tuple(int(dim) for dim in arg1_1.shape)
    flat_shape = _shape_tuple(shape_flat)
    sum_shape = _shape_tuple(shape_sum_bf16)
    rows = int(flat_shape[0])
    channels = int(flat_shape[1])
    device = arg1_1.device

    # Pad to power-of-2 block width (1024) for BLOCK_C
    padded_c = 1024
    assert padded_c >= channels

    # Pre-permute arg3_1 [32, 768, 512] to [32, 512, 768] contiguous
    arg3_perm = arg3_1.permute(0, 2, 1).contiguous().view(rows, channels)

    def _pad_bf16(t, extra_c):
        z = torch.zeros((rows, extra_c), device=device, dtype=torch.bfloat16)
        return torch.cat([t.view(rows, channels), z], dim=1).contiguous()

    def _pad_f32(t, extra_c):
        z = torch.zeros((rows, extra_c), device=device, dtype=torch.float32)
        return torch.cat([t.view(rows, channels), z], dim=1).contiguous()

    def _pad_bool(t, extra_c):
        z = torch.zeros((rows, extra_c), device=device, dtype=torch.bool)
        return torch.cat([t.view(rows, channels), z], dim=1).contiguous()

    extra = padded_c - channels
    arg0_padded = _pad_bf16(arg0_1, extra)
    arg1_padded = _pad_f32(arg1_1, extra)
    arg2_padded = _pad_bf16(arg2_1, extra)
    arg3_padded = _pad_bf16(arg3_perm, extra)
    arg4_padded = _pad_bf16(arg4_1, extra)
    arg5_padded = _pad_bf16(arg5_1, extra)
    rhs_padded = _pad_f32(arg7_1, extra)
    mask_padded = _pad_bool(arg9_1, extra)

    weight_padded = torch.zeros((padded_c,), device=device, dtype=torch.float32)
    weight_padded[:channels] = arg6_1

    scale_1d = arg8_1.view(rows)

    grad_out_padded = torch.zeros((rows, padded_c), device=device, dtype=torch.float32)
    bf16_out_padded = torch.zeros((rows, padded_c), device=device, dtype=torch.bfloat16)
    out_x_rhs = torch.zeros((channels,), device=device, dtype=torch.float32)
    out_x = torch.zeros((channels,), device=device, dtype=torch.float32)
    out_bf16_sum_acc = torch.zeros((channels,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _ln_backward_row_kernel,
        (arg0_padded, arg1_padded, arg2_padded, arg3_padded, arg4_padded, arg5_padded,
         weight_padded, rhs_padded, scale_1d, mask_padded,
         grad_out_padded, bf16_out_padded, out_x_rhs, out_x, out_bf16_sum_acc,
         channels, ROW_FACTOR, DROP_SCALE, padded_c),
    )

    # Narrow back to full shapes
    grad_out = grad_out_padded[:, :channels].reshape(full_shape).contiguous()
    bf16_out = bf16_out_padded[:, :channels].contiguous()

    # Final bf16 rounding for the sum_5 output
    out_bf16_sum = out_bf16_sum_acc.to(torch.bfloat16).to(torch.float32)

    return grad_out, out_x_rhs, out_x, bf16_out, bf16_out.permute(1, 0), out_bf16_sum
