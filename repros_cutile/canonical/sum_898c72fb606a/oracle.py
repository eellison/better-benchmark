"""cuTile port of sum_898c72fb606a: Visformer bf16 softmax-backward row kernel.

Replaces the Triton `_visformer_softmax_backward_49_kernel` with three cuTile
kernels (recompute-probs, split-mul, add-and-scale-out) plus a host-side
row-sum via torch.sum. The Triton oracle uses inline `fma.rn.f32` PTX and
explicit bf16 rounding — cuTile is RTNE by default so those become plain
`+`, `*`, `ct.astype(x, ct.bfloat16)`.

Two numerical fidelity issues had to be worked around vs a naive one-kernel
port:

1. The row-sum reduction. Eager and cuTile's `ct.sum` disagree by ~1 ULP for
   some rows (different reduction trees); we sidestep this by computing
   probs/product in cuTile, storing to buffers, and running `torch.sum` on
   the host — which matches the eager `sum.dim_IntList([-1])` semantics
   bit-for-bit.

2. The `-probs * row_sum + product` step. cuTile compiles this to a single
   FMA (one rounding), while eager's `prims.fma` on GPU is not truly fused
   — it does mul then add (two roundings). At extreme corner-case inputs
   (row_denom ~ 1e-3 -> probs ~ 1e3) the difference crosses a bf16 rounding
   boundary. We split the mul and add across two cuTile kernels so the FMA
   optimizer cannot fuse them.

Non-power-of-2 row width (49) is handled via `padding_mode=ct.PaddingMode.ZERO`
tiles of size 64 and a final `torch.narrow` to the valid region.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.08838834764831845
ROWS = 37632   # 128 * 6 * 49
Q = 49
PAD_N = 64


@ct.kernel
def _recompute_probs_product_kernel(
    grad_ptr,             # bf16 [ROWS, Q]
    logits_ptr,           # bf16 [768, Q, Q] strided view of arg1_1
    row_shift_true_ptr,   # f32  [ROWS]
    row_shift_false_ptr,  # f32  [ROWS]
    branch_ptr,           # b8   [ROWS]
    row_denom_ptr,        # f32  [ROWS]
    probs_out_ptr,        # f32  [ROWS, PAD_N]
    product_out_ptr,      # f32  [ROWS, PAD_N]
    Q_C: ct.Constant[int],
    PAD_N_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    flat_bh = row // Q_C
    q_idx = row - flat_bh * Q_C

    grad = ct.load(
        grad_ptr, index=(row, 0), shape=(1, PAD_N_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits_3d = ct.load(
        logits_ptr, index=(flat_bh, q_idx, 0), shape=(1, 1, PAD_N_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits = ct.reshape(logits_3d, (1, PAD_N_C))
    rst = ct.load(row_shift_true_ptr, index=(row,), shape=(1,))
    rsf = ct.load(row_shift_false_ptr, index=(row,), shape=(1,))
    branch = ct.load(branch_ptr, index=(row,), shape=(1,))
    rd = ct.load(row_denom_ptr, index=(row,), shape=(1,))

    grad_f = ct.astype(grad, ct.float32)
    logits_f = ct.astype(logits, ct.float32)

    logits_scaled_bf16 = ct.astype(logits_f * SCALE_C, ct.bfloat16)
    logits_scaled_bf16_f = ct.astype(logits_scaled_bf16, ct.float32)

    rst_2d = ct.reshape(rst, (1, 1))
    rsf_2d = ct.reshape(rsf, (1, 1))
    branch_2d = ct.reshape(branch, (1, 1))
    rd_2d = ct.reshape(rd, (1, 1))

    branch_scaled_sub = (logits_f - rst_2d) * SCALE_C
    branch_sub_scaled = logits_scaled_bf16_f - rsf_2d
    exp_arg = ct.where(branch_2d, branch_scaled_sub, branch_sub_scaled)
    probs = ct.exp(exp_arg) / rd_2d
    product = grad_f * probs

    ct.store(probs_out_ptr, index=(row, 0), tile=probs)
    ct.store(product_out_ptr, index=(row, 0), tile=product)


@ct.kernel
def _neg_probs_mul_sum_kernel(
    probs_ptr,       # f32 [ROWS, PAD_N]
    row_sum_ptr,     # f32 [ROWS]
    out_ptr,         # f32 [ROWS, PAD_N]  (= -probs * row_sum, no FMA with product)
    PAD_N_C: ct.Constant[int],
):
    row = ct.bid(0)
    probs = ct.load(probs_ptr, index=(row, 0), shape=(1, PAD_N_C))
    row_sum = ct.load(row_sum_ptr, index=(row,), shape=(1,))
    row_sum_2d = ct.reshape(row_sum, (1, 1))
    result = -probs * row_sum_2d
    ct.store(out_ptr, index=(row, 0), tile=result)


@ct.kernel
def _final_epilogue_kernel(
    neg_probs_scaled_ptr,  # f32 [ROWS, PAD_N]
    product_ptr,           # f32 [ROWS, PAD_N]
    out_ptr,               # bf16 [ROWS, PAD_N]
    PAD_N_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    nprs = ct.load(neg_probs_scaled_ptr, index=(row, 0), shape=(1, PAD_N_C))
    prod = ct.load(product_ptr, index=(row, 0), shape=(1, PAD_N_C))
    fma_f = nprs + prod
    rounded_bf16 = ct.astype(fma_f, ct.bfloat16)
    scaled = ct.astype(rounded_bf16, ct.float32) * SCALE_C
    out_bf16 = ct.astype(scaled, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out_bf16)


@oracle_impl(hardware="B200", point="cdec8791")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0, _shape_param_1

    device = arg0_1.device

    grad_2d = arg0_1.reshape(ROWS, Q)
    logits_view = arg1_1[:, :Q, :Q]
    row_shift_true_1d = arg2_1.reshape(ROWS)
    row_shift_false_1d = arg3_1.reshape(ROWS)
    branch_1d = arg4_1.reshape(ROWS)
    row_denom_1d = arg5_1.reshape(ROWS)

    probs_buf = torch.empty((ROWS, PAD_N), device=device, dtype=torch.float32)
    product_buf = torch.empty((ROWS, PAD_N), device=device, dtype=torch.float32)
    neg_probs_scaled_buf = torch.empty(
        (ROWS, PAD_N), device=device, dtype=torch.float32,
    )
    out_padded = torch.empty((ROWS, PAD_N), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _recompute_probs_product_kernel,
        (
            grad_2d, logits_view,
            row_shift_true_1d, row_shift_false_1d,
            branch_1d, row_denom_1d,
            probs_buf, product_buf,
            Q, PAD_N, SCALE,
        ),
    )
    # Row sum via torch.sum matches eager's `sum.dim_IntList([-1])` bit-for-bit.
    # Only cols [0, Q) contribute — grad is zero at padded cols so product is
    # zero there too.
    row_sum = product_buf[:, :Q].sum(dim=1)  # (ROWS,)

    # Compute -probs * row_sum in a separate kernel so it cannot be FMA'd
    # into the subsequent add — that FMA would flip a bf16 rounding boundary
    # at extreme corner-case rows (row_denom ~ 1e-3).
    ct.launch(
        stream, (ROWS, 1, 1), _neg_probs_mul_sum_kernel,
        (probs_buf, row_sum, neg_probs_scaled_buf, PAD_N),
    )
    ct.launch(
        stream, (ROWS, 1, 1), _final_epilogue_kernel,
        (neg_probs_scaled_buf, product_buf, out_padded, PAD_N, SCALE),
    )

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = out_padded[:, :Q].contiguous().view(out_shape)
    return out
