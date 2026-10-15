"""cuTile port of sum_sum_sum_0e08b9d50286: Longformer masked atomic scatter.

Uses cuTile to compute the pointwise per-row LN-backward-like scaled tensor,
then torch for the atomic scatter (index_put_) into three embedding tables
and the per-channel sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 768
POS_BUCKETS = 4098
VOCAB = 50265


@ct.kernel
def _ln_backward_scale_kernel(
    dropout_mask_ptr,   # b8 [rows, BLOCK_H]
    grad_ptr,           # f32 [rows, BLOCK_H]
    weight_ptr,         # f32 [BLOCK_H]
    orig_x_ptr,         # f32 [rows, BLOCK_H]
    invstd_ptr,         # f32 [rows] (arg4_1 broadcast)
    out_ptr,            # f32 [rows, BLOCK_H]
    out_mul1_ptr,       # f32 [rows, BLOCK_H]  (mul_1 = grad * dropout_scaled)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    mask = ct.load(dropout_mask_ptr, index=(row, 0), shape=(1, BLOCK_H))
    grad = ct.load(grad_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    x = ct.load(orig_x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))

    mask_f = ct.astype(mask, ct.float32) * 1.1111111111111112
    mul_1 = grad * mask_f
    ct.store(out_mul1_ptr, index=(row, 0), tile=mul_1)
    mul_2 = mul_1 * weight_2d
    mul_3 = mul_2 * HIDDEN_C
    # sum_2 (row_dot) computed in-kernel to match Triton's row_dot reduction.
    sum_2 = ct.sum(mul_2 * x, axis=1, keepdims=True)
    sum_1 = ct.sum(mul_2, axis=1, keepdims=True)
    invstd_2d = ct.reshape(invstd, (1, 1))
    mul_5 = x * sum_2
    sub = mul_3 - sum_1
    sub_1 = sub - mul_5
    out_val = invstd_2d * sub_1
    ct.store(out_ptr, index=(row, 0), tile=out_val)


@ct.kernel
def _hidden_reduce_kernel(
    mul1_ptr,      # f32 [ROWS, BLOCK_H_pad]
    x_ptr,         # f32 [ROWS, BLOCK_H_pad]
    sum3_ptr,      # f32 [HIDDEN]
    sum4_ptr,      # f32 [HIDDEN]
    ROWS_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    mul1 = ct.load(mul1_ptr, index=(0, col_block), shape=(ROWS_, BLOCK_C))
    x = ct.load(x_ptr, index=(0, col_block), shape=(ROWS_, BLOCK_C))
    prod = mul1 * x
    s3 = ct.sum(prod, axis=0)
    s4 = ct.sum(mul1, axis=0)
    ct.store(sum3_ptr, index=(col_block,), tile=s3)
    ct.store(sum4_ptr, index=(col_block,), tile=s4)


@oracle_impl(hardware="B200", point="05376402", BLOCK_H=768)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
        *_shape_params,
    ) = inputs
    device = arg1_1.device

    # Run cuTile kernel to compute LN-backward scaled tensor mul_6 and (as
    # side effect) store mul_1 for later scatter. Row reductions (sum_1,
    # sum_2) happen in-kernel to match Triton's row_sum/row_dot.
    BLOCK_H_pad = 1024
    n_rows = BATCH * SEQ  # 8192

    dropout_pad = torch.zeros((n_rows, BLOCK_H_pad), device=device, dtype=torch.bool)
    grad_pad = torch.zeros((n_rows, BLOCK_H_pad), device=device, dtype=torch.float32)
    weight_pad = torch.zeros((BLOCK_H_pad,), device=device, dtype=torch.float32)
    x_pad = torch.zeros((n_rows, BLOCK_H_pad), device=device, dtype=torch.float32)
    out_pad = torch.empty((n_rows, BLOCK_H_pad), device=device, dtype=torch.float32)
    mul1_pad = torch.empty((n_rows, BLOCK_H_pad), device=device, dtype=torch.float32)

    dropout_pad[:, :HIDDEN] = arg0_1.view(n_rows, HIDDEN)
    grad_pad[:, :HIDDEN] = arg1_1.view(n_rows, HIDDEN)
    weight_pad[:HIDDEN] = arg2_1
    x_pad[:, :HIDDEN] = arg3_1.view(n_rows, HIDDEN)
    invstd_1d = arg4_1.view(n_rows).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _ln_backward_scale_kernel,
        (dropout_pad, grad_pad, weight_pad, x_pad, invstd_1d,
         out_pad, mul1_pad, HIDDEN, BLOCK_H_pad),
    )
    mul_6 = out_pad[:, :HIDDEN].contiguous().view(BATCH, SEQ, HIDDEN)
    mul_1 = mul1_pad[:, :HIDDEN].contiguous().view(BATCH, SEQ, HIDDEN)

    # Hidden reductions (sum_3=mul_1*x, sum_4=mul_1) in a cuTile kernel —
    # Triton does these accumulations in-kernel via atomic_add. cuTile does
    # them via an explicit reduction pass.
    sum_3_pad = torch.empty((BLOCK_H_pad,), device=device, dtype=torch.float32)
    sum_4_pad = torch.empty((BLOCK_H_pad,), device=device, dtype=torch.float32)
    HIDDEN_BLOCK_C = 8
    ct.launch(
        stream,
        (BLOCK_H_pad // HIDDEN_BLOCK_C, 1, 1),
        _hidden_reduce_kernel,
        (mul1_pad, x_pad, sum_3_pad, sum_4_pad, n_rows, HIDDEN_BLOCK_C),
    )
    sum_3 = sum_3_pad[:HIDDEN]
    sum_4 = sum_4_pad[:HIDDEN]

    # Now the three atomic scatters via torch's index_put_.
    full_true = torch.ones((BATCH, SEQ, 1), device=device, dtype=torch.bool)
    full_1 = torch.zeros((1, HIDDEN), device=device, dtype=torch.float32)
    ip1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_1, full_true, [arg5_1], mul_6)

    ge = arg6_1 >= 0
    lt = arg6_1 < POS_BUCKETS
    ne = arg6_1 != 1
    mask_pos = (ge & lt & ne).unsqueeze(-1)
    full_2 = torch.zeros((POS_BUCKETS, HIDDEN), device=device, dtype=torch.float32)
    ip2 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_2, mask_pos, [arg6_1], mul_6)

    full_3 = torch.zeros((VOCAB, HIDDEN), device=device, dtype=torch.float32)
    ip3 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(
        full_3, arg7_1, [arg8_1], mul_6)

    return sum_3, sum_4, ip1, ip2, ip3
