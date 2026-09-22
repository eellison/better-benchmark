"""cuTile port of sum_sum_sum_836354f50614: Transformer layer-norm-backward tail.

Reference (2D row-major bf16 inputs [rows, hidden]):
  x       = f32(arg0) + f32(arg1) + f32(arg2)                    f32 [rows, hidden]
  mul     = x * weight                                             f32 [rows, hidden]
  row_sum = sum(mul, dim=-1)                                        f32 [rows]
  sub     = arg4 - shift                                            f32 [rows, hidden]
  scaled  = sub * scale                                             (scale is per-row [rows])
  row_dot = sum(mul * scaled, dim=-1)                               f32 [rows]
  grad    = (scale/HIDDEN) * (mul*HIDDEN - row_sum - scaled*row_dot)
  out     = arg7 + grad                                              f32 [rows, hidden]  (viewed as 3D)
  sum_3   = sum(mul * scaled, [0, 1])                                f32 [hidden]
  sum_4   = sum(x, [0, 1])                                           f32 [hidden]
  Returns (sum_3, sum_4, out).

Actually looking more carefully at the reference:
  sum_3 = sum(mul_6 = x * scaled, [0,1])   (mul_6 is x * mul_2 where mul_2 = sub*scale)
  sum_4 = sum(x, [0, 1])   ← the pre-weight x (add_1)

Port plan: one row-tiled kernel that computes x, row_sum, row_dot,
per-row epilogue grad, per-row partials for hidden-column reductions.
Second kernel finalizes partials to [hidden].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NORM_FACTOR = 2560.0


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


def _make_kernels(ROWS_PER_GROUP, BLOCK_C):

    @ct.kernel
    def _row_kernel(
        x0_ptr,          # bf16 [rows, hidden]
        x1_ptr,          # bf16
        x2_ptr,          # bf16
        weight_ptr,      # f32 [hidden]
        source_ptr,      # f32 [rows, hidden]   (arg4)
        shift_ptr,       # f32 [rows]           (arg5 flattened)
        scale_ptr,       # f32 [rows]           (arg6 flattened)
        residual_ptr,    # f32 [rows, hidden]   (arg7)
        out_ptr,         # f32 [rows, hidden]
        partial_dot_ptr, # f32 [num_groups, hidden]  (for sum_3)
        partial_x_ptr,   # f32 [num_groups, hidden]  (for sum_4)
        ROWS: ct.Constant[int],
        HIDDEN: ct.Constant[int],
        NORM_FACTOR_: ct.Constant[float],
    ):
        group = ct.bid(0)
        weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                          padding_mode=ct.PaddingMode.ZERO)
        cols = ct.arange(BLOCK_C, dtype=ct.int32)
        col_mask = cols < HIDDEN

        acc_dot = ct.zeros((BLOCK_C,), dtype=ct.float32)
        acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)

        for local_row in ct.static_iter(range(ROWS_PER_GROUP)):
            row = group * ROWS_PER_GROUP + local_row
            row_active = row < ROWS

            x0 = ct.load(x0_ptr, index=(row, 0), shape=(1, BLOCK_C),
                          padding_mode=ct.PaddingMode.ZERO)
            x1 = ct.load(x1_ptr, index=(row, 0), shape=(1, BLOCK_C),
                          padding_mode=ct.PaddingMode.ZERO)
            x2 = ct.load(x2_ptr, index=(row, 0), shape=(1, BLOCK_C),
                          padding_mode=ct.PaddingMode.ZERO)
            source = ct.load(source_ptr, index=(row, 0), shape=(1, BLOCK_C),
                              padding_mode=ct.PaddingMode.ZERO)
            residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_C),
                                padding_mode=ct.PaddingMode.ZERO)
            shift = ct.load(shift_ptr, index=(row,), shape=(1,),
                             padding_mode=ct.PaddingMode.ZERO)
            scale = ct.load(scale_ptr, index=(row,), shape=(1,),
                             padding_mode=ct.PaddingMode.ZERO)

            x = ct.astype(x0, ct.float32) + ct.astype(x1, ct.float32) + \
                ct.astype(x2, ct.float32)
            weight_bc = ct.broadcast_to(ct.reshape(weight, (1, BLOCK_C)),
                                          (1, BLOCK_C))
            mul_v = x * weight_bc
            shift_bc = ct.broadcast_to(ct.reshape(shift, (1, 1)), (1, BLOCK_C))
            scale_bc = ct.broadcast_to(ct.reshape(scale, (1, 1)), (1, BLOCK_C))
            sub_v = source - shift_bc
            scaled = sub_v * scale_bc

            col_mask_bc = ct.reshape(col_mask, (1, BLOCK_C))
            active_mask = ct.broadcast_to(col_mask_bc, (1, BLOCK_C))
            mul_masked = ct.where(active_mask, mul_v, 0.0)
            weighted_scaled = mul_v * scaled
            weighted_scaled_masked = ct.where(active_mask, weighted_scaled, 0.0)
            row_sum = ct.sum(mul_masked, axis=1)          # (1,)
            row_dot = ct.sum(weighted_scaled_masked, axis=1)  # (1,)
            row_sum_bc = ct.broadcast_to(ct.reshape(row_sum, (1, 1)),
                                          (1, BLOCK_C))
            row_dot_bc = ct.broadcast_to(ct.reshape(row_dot, (1, 1)),
                                          (1, BLOCK_C))

            centered_grad = mul_v * NORM_FACTOR_ - row_sum_bc - scaled * row_dot_bc
            grad = (scale_bc / NORM_FACTOR_) * centered_grad
            out_row = residual + grad
            if row_active:
                ct.store(out_ptr, index=(row, 0), tile=out_row)

            acc_dot = acc_dot + ct.where(col_mask, ct.reshape(x * scaled,
                                                                 (BLOCK_C,)),
                                           0.0)
            acc_x = acc_x + ct.where(col_mask, ct.reshape(x, (BLOCK_C,)), 0.0)

        ct.store(partial_dot_ptr, index=(group, 0),
                  tile=ct.reshape(acc_dot, (1, BLOCK_C)))
        ct.store(partial_x_ptr, index=(group, 0),
                  tile=ct.reshape(acc_x, (1, BLOCK_C)))

    return _row_kernel


# Preinstantiate kernels for each shape.
# 13570cda: rows=4096 hidden=2560  full_shape=(32,128,2560)
# 637b22e7: rows=2048 hidden=2560  full_shape=(16,128,2560)
# c0772b17: rows=8192 hidden=1024  full_shape=(64,128,1024)
# 1cfd796c: rows=8192 hidden=1024  full_shape=(8,1024,1024)
# 2805f48b: rows=8192 hidden=768   full_shape=(4,2048,768)
# 83be941c: rows=16384 hidden=1024 full_shape=(128,128,1024)
# 1247fcfd: rows=4096 hidden=1024  full_shape=(32,128,1024)

_K_2560 = _make_kernels(ROWS_PER_GROUP=8, BLOCK_C=4096)
_K_1024 = _make_kernels(ROWS_PER_GROUP=16, BLOCK_C=1024)
_K_768 = _make_kernels(ROWS_PER_GROUP=16, BLOCK_C=1024)


def _run(inputs, *, ROWS_PER_GROUP, BLOCK_C, kernel):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        shape_param_0, _shape_param_1, _shape_param_2,
    ) = inputs
    full_shape = tuple(int(x) for x in shape_param_0)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    num_groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP

    # Flatten arg5, arg6 (both [B, T, 1] -> [rows])
    shift_1d = arg5_1.contiguous().view(rows)
    scale_1d = arg6_1.contiguous().view(rows)
    source_2d = arg4_1.contiguous().view(rows, hidden)
    residual_2d = arg7_1.contiguous().view(rows, hidden)

    out_2d = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_groups, BLOCK_C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_groups, BLOCK_C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_groups, 1, 1), kernel,
        (arg0_1, arg1_1, arg2_1, arg3_1, source_2d, shift_1d, scale_1d,
         residual_2d, out_2d, partial_dot, partial_x,
         rows, hidden, NORM_FACTOR),
    )

    sum_3 = partial_dot[:, :hidden].sum(dim=0)
    sum_4 = partial_x[:, :hidden].sum(dim=0)

    # Reshape out to full_shape strided (bs, T, hidden) with contiguous strides.
    out_3d = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device, dtype=torch.float32,
    )
    out_3d.copy_(out_2d.view(full_shape))

    return sum_3, sum_4, out_3d


# 13570cda: hidden=2560
@oracle_impl(hardware="B200", point="13570cda")
def oracle_forward_13570cda(inputs):
    return _run(inputs, ROWS_PER_GROUP=8, BLOCK_C=4096, kernel=_K_2560)


# 637b22e7: hidden=2560
@oracle_impl(hardware="B200", point="637b22e7")
def oracle_forward_637b22e7(inputs):
    return _run(inputs, ROWS_PER_GROUP=8, BLOCK_C=4096, kernel=_K_2560)


# c0772b17: hidden=1024
@oracle_impl(hardware="B200", point="c0772b17")
def oracle_forward_c0772b17(inputs):
    return _run(inputs, ROWS_PER_GROUP=16, BLOCK_C=1024, kernel=_K_1024)


# 1cfd796c: hidden=1024
@oracle_impl(hardware="B200", point="1cfd796c")
def oracle_forward_1cfd796c(inputs):
    return _run(inputs, ROWS_PER_GROUP=16, BLOCK_C=1024, kernel=_K_1024)


# 2805f48b: hidden=768
@oracle_impl(hardware="B200", point="2805f48b")
def oracle_forward_2805f48b(inputs):
    return _run(inputs, ROWS_PER_GROUP=16, BLOCK_C=1024, kernel=_K_768)


# 83be941c: hidden=1024
@oracle_impl(hardware="B200", point="83be941c")
def oracle_forward_83be941c(inputs):
    return _run(inputs, ROWS_PER_GROUP=16, BLOCK_C=1024, kernel=_K_1024)


# 1247fcfd: hidden=1024
@oracle_impl(hardware="B200", point="1247fcfd")
def oracle_forward(inputs):
    return _run(inputs, ROWS_PER_GROUP=16, BLOCK_C=1024, kernel=_K_1024)
