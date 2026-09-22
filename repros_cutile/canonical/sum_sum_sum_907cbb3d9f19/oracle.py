"""cuTile port of sum_sum_sum_907cbb3d9f19: GoogleFnet LN-backward with
select_scatter and atomic dgamma/dbeta accumulation.

Per row (of the 16384 = 32*512):
  add = mm + residual
  weighted = add * gamma
  row_sum = sum_c(weighted); row_dot = sum_c(weighted * xhat)
  grad = scale * (weighted * H - row_sum - xhat * row_dot)
  dgamma += add * xhat  (per channel, atomic add)
  dbeta  += add          (per channel, atomic add)
  scatter[..., 0] = grad; scatter[..., 1] = full[..., 1] (unchanged)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ  # 16384
HIDDEN = 768
BLOCK_H = 1024  # padded to next pow2


@ct.kernel
def _ln_backward_row_kernel(
    mm_ptr,        # f32 [ROWS, HIDDEN]
    residual_ptr,  # f32 [ROWS, HIDDEN]
    gamma_ptr,     # f32 [HIDDEN]
    xhat_ptr,      # f32 [ROWS, HIDDEN]
    scale_ptr,     # f32 [ROWS]
    grad_ptr,      # f32 [ROWS, HIDDEN]
    dgamma_ptr,    # f32 [HIDDEN]
    dbeta_ptr,     # f32 [HIDDEN]
    scatter_lane0_ptr,  # f32 [ROWS, HIDDEN]  -- view of scatter[..., 0]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    cols = ct.arange(BLOCK_H_, dtype=ct.int32)
    valid = cols < HIDDEN_
    zero_f = ct.full(shape=(BLOCK_H_,), fill_value=0.0, dtype=ct.float32)

    mm = ct.load(
        mm_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    gamma = ct.load(
        gamma_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    xhat = ct.load(
        xhat_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scale = ct.load(scale_ptr, index=(row,), shape=(1,))
    scale_val = ct.reshape(scale, (1,))  # (1,)

    add = mm + residual  # (1, BLOCK_H)
    add_1d = ct.reshape(add, (BLOCK_H_,))
    xhat_1d = ct.reshape(xhat, (BLOCK_H_,))
    weighted_1d = add_1d * gamma  # (BLOCK_H,)
    weighted_masked = ct.where(valid, weighted_1d, zero_f)

    row_sum = ct.sum(weighted_masked)
    row_dot = ct.sum(weighted_masked * xhat_1d)

    grad_1d = scale_val * (weighted_1d * HIDDEN_F - row_sum - xhat_1d * row_dot)
    grad_masked = ct.where(valid, grad_1d, zero_f)
    grad_2d = ct.reshape(grad_masked, (1, BLOCK_H_))

    ct.store(grad_ptr, index=(row, 0), tile=grad_2d)
    ct.store(scatter_lane0_ptr, index=(row, 0), tile=grad_2d)

    # dgamma += add * xhat, dbeta += add (per channel).
    dg_incr = ct.where(valid, add_1d * xhat_1d, zero_f)
    db_incr = ct.where(valid, add_1d, zero_f)
    # Redirect OOB columns to safe index; atomic_add checks bounds by default.
    ct.atomic_add(dgamma_ptr, (cols,), dg_incr)
    ct.atomic_add(dbeta_ptr, (cols,), db_incr)


@oracle_impl(hardware="B200", point="356087b1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0 = inputs
    device = arg0_1.device

    grad = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    dgamma = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    dbeta = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)

    # Preallocate scatter with same shape/stride as arg5_1, copy from arg5_1 so
    # lane 1 preserves the original data (select_scatter replaces only lane 0).
    scatter = torch.empty_strided(
        tuple(arg5_1.shape),
        tuple(arg5_1.stride()),
        device=device,
        dtype=arg5_1.dtype,
    )
    scatter.copy_(arg5_1)

    # Views for kernel
    mm_2d = arg0_1.view(ROWS, HIDDEN)
    residual_2d = arg1_1.contiguous().view(ROWS, HIDDEN)
    xhat_2d = arg3_1.contiguous().view(ROWS, HIDDEN)
    scale_1d = arg4_1.contiguous().view(ROWS)
    grad_2d = grad.view(ROWS, HIDDEN)
    # scatter is [32, 512, 768, 2]; lane 0 view = scatter[..., 0] -> (32, 512, 768)
    scatter_lane0 = scatter.select(3, 0)  # (32, 512, 768) with strides (…, 2, 2*768)? need contiguous
    scatter_lane0_2d = scatter_lane0.reshape(ROWS, HIDDEN) if scatter_lane0.is_contiguous() else scatter_lane0

    # scatter's strides come from arg5_1. If stride on last dim of arg5_1 is 1
    # (i.e., last-fastest), then scatter[..., 0] is a strided slice. We just
    # pass it directly; cuTile respects strides.
    scatter_lane0_view = scatter.select(3, 0)  # (32, 512, 768), strided

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _ln_backward_row_kernel,
        (mm_2d, residual_2d, arg2_1, xhat_2d, scale_1d,
         grad_2d, dgamma, dbeta, scatter_lane0_view.view(ROWS, HIDDEN),
         HIDDEN, BLOCK_H, float(HIDDEN)),
    )

    return grad, dgamma, dbeta, scatter
