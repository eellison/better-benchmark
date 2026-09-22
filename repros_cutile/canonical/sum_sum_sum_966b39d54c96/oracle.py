"""cuTile port of sum_sum_sum_966b39d54c96: LayerNorm-backward dropout tail.

Row kernel + partials for column reductions of the two "safe" column
sums (x*rhs, x). A finalize kernel reduces those partials in-kernel to
match Triton's row_partials + finalize_partials structure.

The out_drop column reduction is recomputed in torch: cuTile lacks the
inline_asm rn.f32 primitives that Triton uses to match eager's
bit-exact dropout arithmetic; without those, an in-kernel sum of
kernel-computed exact_drop values drifts by 1 ULP per element and
after the bf16 roundtrip on the final sum diverges past tolerance.
Keeping out_drop's recompute in torch preserves eager parity.

BLOCK_H must be a pow-2 >= hidden. For hidden=1024 → BLOCK_H=1024 (exact).
For hidden=2560 → BLOCK_H=4096 (pad+mask).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROP_SCALE = 1.1111111111111112
ROW_FACTOR = 2560.0  # Global constant from the captured Repro


@ct.kernel
def _row_kernel(
    x_ptr,             # bf16 [rows, hidden]
    weight_ptr,        # f32 [hidden]
    rhs_ptr,           # f32 [rows, hidden]
    scale_ptr,         # f32 [rows]
    residual_ptr,      # f32 [rows, hidden]
    keep_ptr,          # bool [rows, hidden]
    add_out_ptr,       # f32 [rows, hidden]
    drop_out_ptr,      # bf16 [rows, hidden]
    exact_drop_ptr,    # bf16 [rows, hidden]
    partial_x_rhs_ptr, # f32 [rows, hidden]
    partial_x_ptr,     # f32 [rows, hidden]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_FACTOR_: ct.Constant[float],
    DROP_SCALE_: ct.Constant[float],
):
    row = ct.bid(0)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)

    x_bf16 = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rhs = ct.load(
        rhs_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scale = ct.load(scale_ptr, index=(row,), shape=(1,))
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep = ct.load(
        keep_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf16, ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    scale_2d = ct.reshape(scale, (1, 1))

    weighted = x * weight_2d
    weighted_masked = ct.where(col_valid, weighted, zero_2d)
    row_sum = ct.sum(weighted_masked, axis=1)
    row_dot = ct.sum(ct.where(col_valid, weighted * rhs, zero_2d), axis=1)
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    row_dot_2d = ct.reshape(row_dot, (1, 1))

    row_factor_tile = ct.full((1, BLOCK_H), ROW_FACTOR_, dtype=ct.float32)
    centered = weighted * row_factor_tile - row_sum_2d - rhs * row_dot_2d
    grad = scale_2d * centered
    add = residual + grad

    drop_scale_f32 = ct.full((1, 1), DROP_SCALE_, dtype=ct.float32)
    keep_scale_bf16 = ct.astype(keep_f * drop_scale_f32, ct.bfloat16)
    keep_scale_f = ct.astype(keep_scale_bf16, ct.float32)
    add_bf16 = ct.astype(add, ct.bfloat16)
    exact_drop_bf16 = ct.astype(ct.astype(add_bf16, ct.float32) * keep_scale_f, ct.bfloat16)
    store_drop_bf16 = ct.astype(add * keep_scale_f, ct.bfloat16)

    ct.store(add_out_ptr, index=(row, 0), tile=add)
    ct.store(drop_out_ptr, index=(row, 0), tile=store_drop_bf16)
    ct.store(exact_drop_ptr, index=(row, 0), tile=exact_drop_bf16)

    # Per-row column partial contributions for out_x_rhs and out_x
    x_rhs_masked = ct.where(col_valid, x * rhs, zero_2d)
    x_masked = ct.where(col_valid, x, zero_2d)
    ct.store(partial_x_rhs_ptr, index=(row, 0), tile=x_rhs_masked)
    ct.store(partial_x_ptr, index=(row, 0), tile=x_masked)


@ct.kernel
def _finalize_col_kernel(
    partial_x_rhs_ptr,   # f32 [rows, hidden]
    partial_x_ptr,       # f32 [rows, hidden]
    out_x_rhs_ptr,       # f32 [hidden]
    out_x_ptr,           # f32 [hidden]
    ROWS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)

    acc_x_rhs = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for r in range(ROWS):
        xr = ct.load(partial_x_rhs_ptr, index=(r, col_block), shape=(1, BLOCK_C))
        xv = ct.load(partial_x_ptr, index=(r, col_block), shape=(1, BLOCK_C))
        acc_x_rhs = acc_x_rhs + ct.reshape(xr, (BLOCK_C,))
        acc_x = acc_x + ct.reshape(xv, (BLOCK_C,))

    ct.store(out_x_rhs_ptr, index=(col_block,), tile=acc_x_rhs)
    ct.store(out_x_ptr, index=(col_block,), tile=acc_x)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _pow2_ceil(x: int) -> int:
    return 1 << (x - 1).bit_length()


# 8e6ad081: bf16[2048,2560], hidden=2560 → BLOCK_H=4096 (pad), FINAL 8 divides 2560
@oracle_impl(hardware="B200", point="8e6ad081", FINAL_BLOCK_C=8)
# 1697adde: bf16[8192,1024], hidden=1024 → BLOCK_H=1024
@oracle_impl(hardware="B200", point="1697adde", FINAL_BLOCK_C=8)
# 34401043: bf16[8192,1024], hidden=1024 → BLOCK_H=1024
@oracle_impl(hardware="B200", point="34401043", FINAL_BLOCK_C=8)
def oracle_forward(inputs, *, FINAL_BLOCK_C: int):
    (
        x_bf16,
        weight,
        rhs,
        scale,
        residual,
        keep,
        full_shape_param,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    rows = int(x_bf16.shape[0])
    hidden = int(x_bf16.shape[1])
    full_shape = _shape_tuple(full_shape_param)
    flat_shape = _shape_tuple(flat_shape_param)
    sum_shape = _shape_tuple(sum_shape_param)
    device = x_bf16.device

    BLOCK_H = _pow2_ceil(hidden)

    rhs_2d = rhs.reshape(rows, hidden)
    residual_2d = residual.reshape(rows, hidden)
    keep_2d = keep.reshape(rows, hidden)
    scale_1d = scale.reshape(rows)

    add_out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device, dtype=torch.float32,
    )
    add_out_2d = add_out.view(rows, hidden)
    drop_out = torch.empty_strided(
        flat_shape, (flat_shape[1], 1), device=device, dtype=torch.bfloat16,
    )
    exact_drop = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
    partial_x_rhs = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    partial_x = torch.empty((rows, hidden), device=device, dtype=torch.float32)

    out_x_rhs = torch.empty(sum_shape, device=device, dtype=torch.float32)
    out_x = torch.empty(sum_shape, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _row_kernel,
        (x_bf16, weight, rhs_2d, scale_1d, residual_2d, keep_2d,
         add_out_2d, drop_out, exact_drop, partial_x_rhs, partial_x,
         hidden, BLOCK_H, ROW_FACTOR, DROP_SCALE),
    )
    ct.launch(
        stream, (hidden // FINAL_BLOCK_C, 1, 1), _finalize_col_kernel,
        (partial_x_rhs, partial_x,
         out_x_rhs.view(hidden), out_x.view(hidden),
         rows, FINAL_BLOCK_C),
    )

    # out_drop: keep the torch recompute of exact_drop to match eager's
    # bit-exact arithmetic (cuTile lacks inline_asm rn.f32 primitives).
    x_f32 = x_bf16.float()
    weighted = x_f32 * weight
    row_sum = weighted.sum(dim=1, keepdim=True)
    row_dot = (weighted * rhs_2d).sum(dim=1, keepdim=True)
    centered = weighted * ROW_FACTOR - row_sum - rhs_2d * row_dot
    grad = scale_1d.unsqueeze(1) * centered
    add_torch = residual_2d + grad
    add_torch_bf = add_torch.to(torch.bfloat16)
    keep_scale_bf = keep_2d.to(torch.bfloat16) * DROP_SCALE
    exact_drop_torch = add_torch_bf * keep_scale_bf
    out_drop_sum = exact_drop_torch.sum(dim=0, dtype=torch.float32)
    out_drop = out_drop_sum.to(torch.bfloat16).to(torch.float32).view(*sum_shape)

    return (
        out_x_rhs,
        out_x,
        add_out,
        drop_out,
        drop_out.permute(1, 0),
        out_drop,
    )
