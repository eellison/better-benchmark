"""cuTile port of sum_sum_sum_d9612b876341: FNet tanh-GELU + LN backward.

Two-kernel structure matching Triton:
  - _row_partials_kernel: per-row LN+GELU backward + partial column sums
  - _finalize_partials_kernel: final column reduction across row-block groups
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
ROW_FACTOR = 768.0
INV_HIDDEN = 1.0 / 768.0
SQRT_2_OVER_PI = 0.7978845608028654
GELU_TANH_CUBE = 0.044715


@ct.kernel
def _row_partials_kernel(
    ln_input_ptr,     # f32 [ROWS, HIDDEN]
    weight_ptr,       # f32 [HIDDEN]
    gelu_input_ptr,   # f32 [ROWS, HIDDEN]
    center_ptr,       # f32 [ROWS]
    row_scale_ptr,    # f32 [ROWS]
    grad_out_ptr,     # f32 [ROWS, HIDDEN]  (final output, not padded)
    partial_x_times_normed_ptr,   # f32 [NUM_GROUPS, BLOCK_C] (padded)
    partial_x_ptr,                # f32 [NUM_GROUPS, BLOCK_C] (padded)
    partial_grad_ptr,             # f32 [NUM_GROUPS, BLOCK_C] (padded)
    HIDDEN_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    ROW_FACTOR_C: ct.Constant[float],
    INV_HIDDEN_C: ct.Constant[float],
    SQRT_2_OVER_PI_C: ct.Constant[float],
    GELU_TANH_CUBE_C: ct.Constant[float],
):
    row = ct.bid(0)

    x = ct.load(ln_input_ptr, index=(row, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    gelu_x = ct.load(gelu_input_ptr, index=(row, 0), shape=(1, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    center = ct.load(center_ptr, index=(row,), shape=(1,))
    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_C))
    zero = ct.full((1, BLOCK_C), 0.0, dtype=ct.float32)

    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    weighted = x * weight_2d
    weighted_masked = ct.where(col_mask_2d, weighted, zero)
    row_sum_val = ct.sum(weighted_masked)

    gelu_half = gelu_x * 0.5
    gelu_x2 = gelu_x * gelu_x
    gelu_x3 = gelu_x2 * gelu_x
    scaled_cubic = gelu_x3 * GELU_TANH_CUBE_C
    gelu_inner = gelu_x + scaled_cubic
    tanh_arg = gelu_inner * SQRT_2_OVER_PI_C
    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    e2x = ct.exp(tanh_arg * 2.0)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    tanh_plus_one = tanh_val + 1.0
    gelu = gelu_half * tanh_plus_one
    center_2d = ct.reshape(center, (1, 1))
    row_scale_2d = ct.reshape(row_scale, (1, 1))
    normed = (gelu - center_2d) * row_scale_2d

    weighted_normed = weighted * normed
    weighted_normed_masked = ct.where(col_mask_2d, weighted_normed, zero)
    row_dot_val = ct.sum(weighted_normed_masked)
    normed_row_dot = normed * row_dot_val

    weighted_times_hidden = weighted * ROW_FACTOR_C
    centered = weighted_times_hidden - row_sum_val - normed_row_dot
    ln_grad = row_scale_2d * INV_HIDDEN_C * centered

    mul11 = ln_grad * gelu_half
    mul12 = ln_grad * tanh_plus_one
    tanh_sq = tanh_val * tanh_val
    sech2 = 1.0 - tanh_sq
    mul14 = mul11 * sech2
    mul15 = mul14 * SQRT_2_OVER_PI_C
    mul16 = mul15 * GELU_TANH_CUBE_C
    mul17 = gelu_x2 * 3.0
    mul18 = mul16 * mul17
    add2 = mul15 + mul18
    mul19 = mul12 * 0.5
    grad = add2 + mul19

    ct.store(grad_out_ptr, index=(row, 0), tile=grad)

    x_times_normed = ct.where(col_mask_2d, x * normed, zero)
    x_masked = ct.where(col_mask_2d, x, zero)
    grad_masked = ct.where(col_mask_2d, grad, zero)
    ct.store(partial_x_times_normed_ptr, index=(row, 0), tile=x_times_normed)
    ct.store(partial_x_ptr, index=(row, 0), tile=x_masked)
    ct.store(partial_grad_ptr, index=(row, 0), tile=grad_masked)


@ct.kernel
def _finalize_partials_kernel(
    partial_x_times_normed_ptr,   # f32 [ROWS, BLOCK_C]
    partial_x_ptr,                # f32 [ROWS, BLOCK_C]
    partial_grad_ptr,             # f32 [ROWS, BLOCK_C]
    out_x_times_normed_ptr,       # f32 [HIDDEN]
    out_x_ptr,                    # f32 [HIDDEN]
    out_grad_sum_ptr,             # f32 [HIDDEN]
    ROWS_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
):
    ht = ct.bid(0)
    # Load a strip of rows of the padded partials of width FINAL_BLOCK_C.
    tile_x = ct.load(partial_x_times_normed_ptr, index=(0, ht),
                     shape=(ROW_BLOCK, FINAL_BLOCK_C))
    tile_y = ct.load(partial_x_ptr, index=(0, ht),
                     shape=(ROW_BLOCK, FINAL_BLOCK_C))
    tile_g = ct.load(partial_grad_ptr, index=(0, ht),
                     shape=(ROW_BLOCK, FINAL_BLOCK_C))
    acc_x = ct.sum(tile_x, axis=0)
    acc_y = ct.sum(tile_y, axis=0)
    acc_g = ct.sum(tile_g, axis=0)
    ct.store(out_x_times_normed_ptr, index=(ht,), tile=acc_x)
    ct.store(out_x_ptr, index=(ht,), tile=acc_y)
    ct.store(out_grad_sum_ptr, index=(ht,), tile=acc_g)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="930536e2",
    ROWS_PER_GROUP=8,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=4,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    device = arg0.device
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])

    padded_xn = torch.empty((rows, BLOCK_C), device=device, dtype=torch.float32)
    padded_x = torch.empty((rows, BLOCK_C), device=device, dtype=torch.float32)
    padded_gs = torch.empty((rows, BLOCK_C), device=device, dtype=torch.float32)
    # We need grad_out at (rows, hidden), but the kernel writes padded_grad
    # only through the reduction path (via partial_grad). Actually, we need
    # a separate grad_out. The Triton oracle stores grad directly. We do that
    # via a plain output tensor of shape (rows, hidden). Since we already
    # have masked writes complex, keep padded_grad as the destination and
    # slice for output.
    padded_grad = torch.empty((rows, BLOCK_C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _row_partials_kernel,
        (arg0, arg1, arg2, arg3.view(rows), arg4.view(rows),
         padded_grad, padded_xn, padded_x, padded_gs,
         hidden, BLOCK_C, ROW_FACTOR, INV_HIDDEN, SQRT_2_OVER_PI, GELU_TANH_CUBE),
    )

    out_x_times_normed = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    out_grad_sum = torch.empty((hidden,), device=device, dtype=torch.float32)

    # Finalize column reductions in-kernel.
    ct.launch(
        stream,
        (hidden // FINAL_BLOCK_C, 1, 1),
        _finalize_partials_kernel,
        (padded_xn, padded_x, padded_gs,
         out_x_times_normed, out_x, out_grad_sum,
         rows, hidden, rows, BLOCK_C, FINAL_BLOCK_C),
    )

    grad_out = padded_grad[:, :hidden].contiguous()
    return out_x_times_normed, out_x, grad_out, grad_out.permute(1, 0), out_grad_sum
