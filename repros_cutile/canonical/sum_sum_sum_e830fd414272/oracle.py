"""cuTile port of sum_sum_sum_e830fd414272: Albert tanh-GELU + LayerNorm backward.

Single-row cooperative approach (HIDDEN=128 fits comfortably):
  1. `_row_kernel`: (ROWS,) grid, one program per row. Load [1, HIDDEN] tile
     of ln_input and gelu_input, compute the tanh-GELU forward algebra plus
     the layer-norm-backward reduction terms with row-local `sum`s, then the
     GELU-backward tail. Store the bf16 grad to grad_out[n, :]. Atomic-add
     per-column contributions of `x * normed`, `x`, and `grad_bf.f32` into
     three [HIDDEN] accumulators.
  2. torch epilogue: `out_grad_sum = accumulator.bf16().f32()` to match the
     Repro's final `bf16 -> f32` cast on the column sum.

All the Triton `add.rn.f32` / `sub.rn.f32` / `mul.rn.f32` inline PTX ops and
`.to(bfloat16, fp_downcast_rounding='rtne')` casts map to cuTile's default
round-to-nearest-even arithmetic and `ct.astype(_, ct.bfloat16)`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
HIDDEN = 128
ROW_FACTOR = 128.0
INV_HIDDEN = 1.0 / 128.0
SQRT_2_OVER_PI = 0.7978845608028654
GELU_TANH_CUBE = 0.044715


@ct.kernel
def _row_kernel(
    ln_input_ptr,      # bf16 [ROWS, HIDDEN]
    weight_ptr,        # f32  [HIDDEN]
    gelu_input_ptr,    # bf16 [ROWS, HIDDEN]
    center_ptr,        # f32  [ROWS]
    row_scale_ptr,     # f32  [ROWS]
    grad_out_ptr,      # bf16 [ROWS, HIDDEN]
    acc_x_normed_ptr,  # f32  [HIDDEN] atomic accumulator
    acc_x_ptr,         # f32  [HIDDEN] atomic accumulator
    acc_grad_ptr,      # f32  [HIDDEN] atomic accumulator
    HIDDEN_: ct.Constant[int],
    ROW_FACTOR_: ct.Constant[float],
    INV_HIDDEN_: ct.Constant[float],
    SQRT_2_OVER_PI_: ct.Constant[float],
    GELU_TANH_CUBE_: ct.Constant[float],
):
    row = ct.bid(0)
    x_bf = ct.load(ln_input_ptr, index=(row, 0), shape=(1, HIDDEN_))
    x = ct.astype(x_bf, ct.float32)
    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))

    gelu_bf = ct.load(gelu_input_ptr, index=(row, 0), shape=(1, HIDDEN_))
    gelu_x = ct.astype(gelu_bf, ct.float32)

    center_v = ct.load(center_ptr, index=(row,), shape=(1,))
    row_scale_v = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    center_2d = ct.reshape(center_v, (1, 1))
    row_scale_2d = ct.reshape(row_scale_v, (1, 1))

    # LN-backward algebra (round-to-nearest-even is cuTile's default).
    weighted = x * weight_2d
    weighted_times_hidden = weighted * ROW_FACTOR_
    row_sum = ct.sum(weighted)  # scalar-like

    # GELU-tanh forward:
    half_x_f = gelu_x * 0.5
    half_x = ct.astype(ct.astype(half_x_f, ct.bfloat16), ct.float32)
    gelu_x2 = gelu_x * gelu_x
    gelu_x3 = gelu_x2 * gelu_x
    scaled_cubic = gelu_x3 * GELU_TANH_CUBE_
    tanh_arg = (gelu_x + scaled_cubic) * SQRT_2_OVER_PI_
    tanh_val = ct.tanh(tanh_arg)
    tanh_plus_one = tanh_val + 1.0
    gelu_forward = half_x * tanh_plus_one

    normed = (gelu_forward - center_2d) * row_scale_2d
    weighted_normed = weighted * normed
    row_dot = ct.sum(weighted_normed)  # scalar

    sub1 = weighted_times_hidden - row_sum
    sub2 = sub1 - normed * row_dot
    ln_grad = (row_scale_2d * INV_HIDDEN_) * sub2

    # GELU-backward tail:
    mul11 = ln_grad * half_x
    mul12 = ln_grad * tanh_plus_one
    mul12_bf = ct.astype(mul12, ct.bfloat16)
    mul12_bf_f = ct.astype(mul12_bf, ct.float32)
    tanh_sq = tanh_val * tanh_val
    sech2 = 1.0 - tanh_sq
    mul15 = mul11 * sech2 * SQRT_2_OVER_PI_
    tail0_bf = ct.astype(mul15, ct.bfloat16)
    tail0_f = ct.astype(tail0_bf, ct.float32)
    mul16 = mul15 * GELU_TANH_CUBE_
    mul17 = gelu_x2 * 3.0
    tail1_bf = ct.astype(mul16 * mul17, ct.bfloat16)
    tail1_f = ct.astype(tail1_bf, ct.float32)
    tail_bf = ct.astype(tail0_f + tail1_f, ct.bfloat16)
    tail_f = ct.astype(tail_bf, ct.float32)
    right_bf = ct.astype(mul12_bf_f * 0.5, ct.bfloat16)
    right_f = ct.astype(right_bf, ct.float32)
    grad_bf = ct.astype(tail_f + right_f, ct.bfloat16)

    ct.store(grad_out_ptr, index=(row, 0), tile=grad_bf)

    grad_f = ct.astype(grad_bf, ct.float32)

    # Atomic-add per-column contributions.
    cols = ct.arange(HIDDEN_, dtype=ct.int32)
    x_1d = ct.reshape(x, (HIDDEN_,))
    normed_1d = ct.reshape(normed, (HIDDEN_,))
    grad_1d = ct.reshape(grad_f, (HIDDEN_,))
    ct.atomic_add(acc_x_normed_ptr, (cols,), x_1d * normed_1d)
    ct.atomic_add(acc_x_ptr, (cols,), x_1d)
    ct.atomic_add(acc_grad_ptr, (cols,), grad_1d)


@oracle_impl(
    hardware="B200",
    point="bcddd3bd",
    ROWS_PER_GROUP=8,
    BLOCK_R=1,
    BLOCK_C=128,
    FINAL_BLOCK_C=16,
    row_warps=4,
    final_warps=8,
)
def oracle_forward(
    inputs, *, ROWS_PER_GROUP, BLOCK_R, BLOCK_C, FINAL_BLOCK_C, row_warps, final_warps,
):
    del ROWS_PER_GROUP, BLOCK_R, BLOCK_C, FINAL_BLOCK_C, row_warps, final_warps
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _s0, _s1, _s2, _s3 = inputs
    device = arg0_1.device

    # arg3_1 and arg4_1 come as [8, 512, 1]; flatten to [ROWS].
    center_1d = arg3_1.view(ROWS)
    row_scale_1d = arg4_1.view(ROWS)

    grad_out = torch.empty_strided(
        (ROWS, HIDDEN), (HIDDEN, 1), device=device, dtype=torch.bfloat16,
    )
    out_x_normed = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    acc_grad = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _row_kernel,
        (arg0_1, arg1_1, arg2_1, center_1d, row_scale_1d, grad_out,
         out_x_normed, out_x, acc_grad,
         HIDDEN, ROW_FACTOR, INV_HIDDEN, SQRT_2_OVER_PI, GELU_TANH_CUBE),
    )

    # Final bf16 round to match Repro's final cast chain.
    out_grad_sum = acc_grad.to(torch.bfloat16).to(torch.float32)

    return out_x_normed, out_x, grad_out, grad_out.permute(1, 0), out_grad_sum
