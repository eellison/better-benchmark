"""cuTile port of sum_24541cd0c55b: XLNet attention-backward row epilogue.

Per row of shape (512,), compute:
  scaled_grad = f32(arg0_row) * (f32(dropout_mask) * 1.1111111111111112)
  true_path = (f32(arg2_row) - arg3_row_scalar) * 0.125         # from mul_3 path
  false_path = f32(bf16(arg2_row * 0.125)) - arg4_row_scalar    # from mul_2 path
  where = arg5_row ? true_path : false_path
  div = exp(where) / arg6_row_scalar
  product = scaled_grad * div
  row_sum = sum(product)
  out = bf16(fma(-div, row_sum, product)) * bf16(0.125)

The 3D view [256, 512, 512] and the [16, 16, 512, 512] shape are both physical
layouts of the same underlying storage (256 = 16 * 16). We fold n_rows =
16*16*512 = 131072 kernel programs, each writing one row (512 cols).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xlnet_attention_backward_kernel(
    grad_ptr,             # bf16 [n_rows, N_COLS]
    dropout_mask_ptr,     # b8 [n_rows, N_COLS]
    prob_ptr,             # bf16 [n_rows, N_COLS]
    row_base_a_ptr,       # f32 [n_rows]
    row_base_b_ptr,       # f32 [n_rows]
    row_pred_ptr,         # b8 [n_rows]
    row_denom_ptr,        # f32 [n_rows]
    out_ptr,              # bf16 [n_rows, N_COLS]
    N_COLS: ct.Constant[int],
):
    row = ct.bid(0)

    grad_bf = ct.load(grad_ptr, index=(row, 0), shape=(1, N_COLS))
    keep_b = ct.load(dropout_mask_ptr, index=(row, 0), shape=(1, N_COLS))
    prob_bf = ct.load(prob_ptr, index=(row, 0), shape=(1, N_COLS))
    base_a = ct.load(row_base_a_ptr, index=(row,), shape=(1,))
    base_b = ct.load(row_base_b_ptr, index=(row,), shape=(1,))
    pred_b = ct.load(row_pred_ptr, index=(row,), shape=(1,))
    denom = ct.load(row_denom_ptr, index=(row,), shape=(1,))

    grad_f = ct.astype(grad_bf, ct.float32)
    keep_f = ct.astype(keep_b, ct.float32)
    prob_f = ct.astype(prob_bf, ct.float32)
    # Note: prob * 0.125 done in bf16 first, then cast to f32
    prob_scaled_bf = ct.astype(prob_bf * ct.astype(0.125, ct.bfloat16), ct.bfloat16)
    prob_scaled_f = ct.astype(prob_scaled_bf, ct.float32)

    base_a_row = ct.reshape(base_a, (1, 1))
    base_b_row = ct.reshape(base_b, (1, 1))
    pred_row = ct.reshape(pred_b, (1, 1))
    denom_row = ct.reshape(denom, (1, 1))

    true_path = (prob_f - base_a_row) * 0.125
    false_path = prob_scaled_f - base_b_row
    # pred_row is b8; use ct.where directly (broadcast to (1, N_COLS))
    exponent = ct.where(pred_row, true_path, false_path)
    div = ct.exp(exponent) / denom_row

    scaled_grad = grad_f * (keep_f * 1.1111111111111112)
    product = scaled_grad * div
    row_sum = ct.sum(product, keepdims=True)
    # fma(-div, row_sum, product) = product - div * row_sum
    fma = product - div * row_sum
    out_bf = ct.astype(fma, ct.bfloat16)
    out_scaled = ct.astype(
        ct.astype(out_bf, ct.float32) * 0.125, ct.bfloat16
    )
    ct.store(out_ptr, index=(row, 0), tile=out_scaled)


@oracle_impl(hardware="B200", point="66f209c9")
def oracle_forward(inputs, **kwargs):
    (
        arg0_1,      # bf16 [256, 512, 512]
        arg1_1,      # b8   [16, 16, 512, 512]
        arg2_1,      # bf16 [16, 16, 512, 512]
        arg3_1,      # f32  [16, 16, 512, 1]
        arg4_1,      # f32  [16, 16, 512, 1]
        arg5_1,      # b8   [16, 16, 512, 1]
        arg6_1,      # f32  [16, 16, 512, 1]
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    # Both physical layouts share the same [16*16*512, 512] row structure.
    n_cols = 512
    # arg0 is [256, 512, 512]. View arg0 -> [16, 16, 512, 512] -> flat rows.
    grad = arg0_1.view(16, 16, 512, 512).view(-1, n_cols)  # [131072, 512]
    dropout_mask = arg1_1.view(-1, n_cols)
    prob = arg2_1.view(-1, n_cols)
    row_base_a = arg3_1.view(-1)
    row_base_b = arg4_1.view(-1)
    row_pred = arg5_1.view(-1)
    row_denom = arg6_1.view(-1)

    n_rows = grad.shape[0]

    out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    out_flat = out.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xlnet_attention_backward_kernel,
        (grad, dropout_mask, prob, row_base_a, row_base_b, row_pred, row_denom,
         out_flat, n_cols),
    )

    # view_2: view [16, 16, 512, 512, 1] -> permute [0,1,2,4,3] -> view [256, 512, 512]
    view_2 = out.view(16, 16, 512, 512, 1).permute(0, 1, 2, 4, 3).contiguous().view(
        *[int(d) for d in _shape_param_2]
    )
    return out, view_2
