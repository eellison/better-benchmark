"""cuTile port of sum_sum_sum_0c3a6fa20cd5: BEiT class-token LN-backward.

Two cuTile kernels:
  1. Row LN-backward kernel: per row (batch), compute row_sum/row_dot of
     weighted x. Also atomic-add sum_3 (=sum(x*norm)) and sum_4 (=sum(x))
     over batch (i.e., column reductions over rows).
     Store the per-row LN-backward result 'row_grad' [BATCH, HIDDEN].
  2. Materialize kernel: for each row (batch), compute divide-by-196 and
     write slice_scatter output at token indices 1..197. Also compute
     projected bf16 and atomic-add sum_5-like accumulator (via bf16 rounding).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
PATCH_TOKENS = 196
HIDDEN = 768
BLOCK_H = 1024  # power of 2 >= HIDDEN


@ct.kernel
def _row_ln_backward_kernel(
    x_ptr,           # bf16 [BATCH, BLOCK_H] (padded)
    gamma_ptr,       # f32  [BLOCK_H] (padded)
    norm_ptr,        # f32  [BATCH, BLOCK_H] (padded)
    inv_scale_ptr,   # f32  [BATCH]
    out_x_norm_ptr,  # f32  [HIDDEN] atomic accumulator
    out_x_ptr,       # f32  [HIDDEN] atomic accumulator
    row_grad_ptr,    # f32  [BATCH, BLOCK_H] (padded, HIDDEN valid)
    HIDDEN_: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.astype(ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_H,))
    norm = ct.load(norm_ptr, index=(row, 0), shape=(1, BLOCK_H))
    inv_scale = ct.load(inv_scale_ptr, index=(row,), shape=(1,))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN_
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    weighted = x * ct.reshape(gamma, (1, BLOCK_H))
    weighted_masked = ct.where(col_valid_2d, weighted, 0.0)
    row_sum = ct.sum(weighted_masked, axis=1, keepdims=True)
    weighted_norm = weighted * norm
    weighted_norm_masked = ct.where(col_valid_2d, weighted_norm, 0.0)
    row_dot = ct.sum(weighted_norm_masked, axis=1, keepdims=True)

    inv_scale_2d = ct.reshape(inv_scale, (1, 1))
    grad = inv_scale_2d * (weighted * float(HIDDEN) - row_sum - norm * row_dot)
    grad_masked = ct.where(col_valid_2d, grad, 0.0)
    ct.store(row_grad_ptr, index=(row, 0), tile=grad_masked)

    # Atomic per-column accumulators
    x_flat = ct.reshape(x, (BLOCK_H,))
    norm_flat = ct.reshape(norm, (BLOCK_H,))
    invalid_cols = ct.full((BLOCK_H,), HIDDEN_, dtype=ct.int32)
    cols_safe = ct.where(col_valid, cols, invalid_cols)
    ct.atomic_add(out_x_norm_ptr, (cols_safe,), x_flat * norm_flat)
    ct.atomic_add(out_x_ptr, (cols_safe,), x_flat)


@ct.kernel
def _materialize_kernel(
    row_grad_ptr,     # f32 [BATCH, BLOCK_H]
    proj_weight_ptr,  # f32 [BLOCK_H] (padded gamma_proj = arg4_1)
    other_ptr,        # bf16 [BATCH, TOKENS, BLOCK_H] (padded)
    scatter_out_ptr,  # f32  [BATCH, TOKENS, BLOCK_H] (padded)
    bf16_out_ptr,     # bf16 [BATCH, TOKENS, BLOCK_H] (padded)
    out_other_ptr,    # f32 [HIDDEN] atomic
    out_bf16_sum_ptr, # f32 [HIDDEN] atomic
    HIDDEN_: ct.Constant[int],
    TOKENS_: ct.Constant[int],
):
    batch = ct.bid(0)
    token = ct.bid(1)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN_
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    grad = ct.load(row_grad_ptr, index=(batch, 0), shape=(1, BLOCK_H))
    proj_weight = ct.load(proj_weight_ptr, index=(0,), shape=(BLOCK_H,))
    other = ct.astype(
        ct.load(other_ptr, index=(batch, token, 0), shape=(1, 1, BLOCK_H)),
        ct.float32,
    )

    # Only token > 0 (patch) contributes to the scatter, and value is grad / 196
    is_patch = token > 0
    patch_value = grad / float(PATCH_TOKENS)
    scatter_value = ct.where(is_patch, patch_value, ct.zeros((1, BLOCK_H), dtype=ct.float32))
    scatter_masked = ct.where(col_valid_2d, scatter_value, 0.0)
    # Store as 3D (1, 1, BLOCK_H)
    scatter_3d = ct.reshape(scatter_masked, (1, 1, BLOCK_H))
    ct.store(scatter_out_ptr, index=(batch, token, 0), tile=scatter_3d)

    projected = scatter_value * ct.reshape(proj_weight, (1, BLOCK_H))
    projected_bf = ct.astype(projected, ct.bfloat16)
    projected_bf_3d = ct.reshape(projected_bf, (1, 1, BLOCK_H))
    ct.store(bf16_out_ptr, index=(batch, token, 0), tile=projected_bf_3d)

    # Column reductions via atomic_add per-column (only across [batch, token])
    other_prod = scatter_value * ct.reshape(other, (1, BLOCK_H))
    other_prod_masked = ct.where(col_valid_2d, other_prod, 0.0)
    projected_bf_f = ct.astype(projected_bf, ct.float32)
    projected_bf_masked = ct.where(col_valid_2d, projected_bf_f, 0.0)

    other_flat = ct.reshape(other_prod_masked, (BLOCK_H,))
    proj_flat = ct.reshape(projected_bf_masked, (BLOCK_H,))

    invalid = ct.full((BLOCK_H,), HIDDEN_, dtype=ct.int32)
    cols_safe = ct.where(col_valid, cols, invalid)
    ct.atomic_add(out_other_ptr, (cols_safe,), other_flat)
    ct.atomic_add(out_bf16_sum_ptr, (cols_safe,), proj_flat)


@oracle_impl(hardware="B200", point="6da1d727", token_warps=4, final_warps=8)
def oracle_forward(inputs, *, token_warps=None, final_warps=None):
    del token_warps, final_warps
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        _shape_param_0, shape_param_1, shape_param_2,
        _shape_param_3, shape_param_4, _shape_param_5,
    ) = inputs
    device = arg0_1.device

    # Pad inputs to BLOCK_H
    def _pad_bf16(t, rows, extra_c):
        z = torch.zeros((rows, extra_c), device=device, dtype=torch.bfloat16)
        return torch.cat([t.view(rows, HIDDEN), z], dim=1).contiguous()

    def _pad_f32(t, rows, extra_c):
        z = torch.zeros((rows, extra_c), device=device, dtype=torch.float32)
        return torch.cat([t.view(rows, HIDDEN), z], dim=1).contiguous()

    extra = BLOCK_H - HIDDEN
    x_padded = _pad_bf16(arg0_1, BATCH, extra)
    norm_padded = _pad_f32(arg2_1, BATCH, extra)
    gamma_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    gamma_padded[:HIDDEN] = arg1_1
    proj_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    proj_padded[:HIDDEN] = arg4_1
    inv_scale = arg3_1.view(BATCH)

    # arg5_1: bf16 [25216, 768] = [BATCH*TOKENS, HIDDEN]; view -> [BATCH, TOKENS, HIDDEN]; pad channel dim
    other_3d = arg5_1.view(BATCH, TOKENS, HIDDEN)
    z_other = torch.zeros((BATCH, TOKENS, extra), device=device, dtype=torch.bfloat16)
    other_padded = torch.cat([other_3d, z_other], dim=2).contiguous()

    # row_grad buffer
    row_grad_padded = torch.zeros((BATCH, BLOCK_H), device=device, dtype=torch.float32)

    out_x_norm = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (BATCH, 1, 1), _row_ln_backward_kernel,
        (x_padded, gamma_padded, norm_padded, inv_scale,
         out_x_norm, out_x, row_grad_padded, HIDDEN),
    )

    # Materialize: scatter_out shape [BATCH, TOKENS, HIDDEN] as f32
    scatter_padded = torch.zeros((BATCH, TOKENS, BLOCK_H), device=device, dtype=torch.float32)
    bf16_padded = torch.zeros((BATCH, TOKENS, BLOCK_H), device=device, dtype=torch.bfloat16)
    out_other = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    out_bf16_sum_acc = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)

    ct.launch(
        stream, (BATCH, TOKENS, 1), _materialize_kernel,
        (row_grad_padded, proj_padded, other_padded, scatter_padded, bf16_padded,
         out_other, out_bf16_sum_acc, HIDDEN, TOKENS),
    )

    # Multiply by proj_weight (arg4_1) to get scatter_scatter with proj (but that's what
    # projected is). scatter_out is `slice_scatter` output — from the eager, it's simply
    # zeros with rows 1..197 filled with div. The `slice_scatter` output is that raw f32.
    scatter_out = scatter_padded[:, :, :HIDDEN].contiguous()

    # bf16_out is `view_2 = view(mul_6, [25216, 768])` where mul_6 = slice_scatter * arg4_1
    # Which is our `bf16_padded`; sliced to HIDDEN and viewed as [BATCH*TOKENS, HIDDEN]
    bf16_out = bf16_padded[:, :, :HIDDEN].reshape(BATCH * TOKENS, HIDDEN).contiguous()

    # For sum_6 (arg5_1 view -> bf16 * ... -> sum), we need the bf16 sum of bf16_out.
    # Our accumulator has f32 sum of projected_bf per column; then final bf16 round.
    out_bf16_sum = out_bf16_sum_acc.to(torch.bfloat16).to(torch.float32)

    return (out_x_norm, out_x, scatter_out, out_other, bf16_out,
            bf16_out.permute(1, 0), out_bf16_sum)
