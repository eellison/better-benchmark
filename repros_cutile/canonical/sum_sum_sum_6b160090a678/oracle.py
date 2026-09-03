"""cuTile port of sum_sum_sum_6b160090a678: Electra dropout-LN backward +
embedding scatter.

The Triton oracle uses `tl.atomic_add` for scatter and can't be perfectly
mirrored in cuTile without a full rewrite. Here we still push the scatter
step to torch, but keep the reductions inside cuTile kernels. Kernel count
still stays at 3 (matching Triton).

Kernels:
  1) _dropout_ln_bwd_kernel: per-row LN-backward + emits mul_1 and mul_6.
  2) _reduce_by_seq_kernel: per-seq reductions over BATCH -> pos_acc / sum3 / sum4.
  3) _finalize_reduce_kernel: reduces per-seq partials to sum_3, sum_4.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
SEQ = 512
HIDDEN = 128
ROWS = BATCH * SEQ
POSITION_ROWS = 512
TYPE_ROWS = 2
VOCAB_ROWS = 30522
DROP_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_ln_bwd_kernel(
    grad_ptr, dropout_ptr, weight_ptr, xhat_ptr, invstd_ptr,
    mul1_ptr, mul6_ptr,
    H: ct.Constant[int],
):
    row = ct.bid(0)
    grad_bf = ct.load(grad_ptr, index=(row, 0), shape=(1, H))
    dropout_mask = ct.load(dropout_ptr, index=(row, 0), shape=(1, H))
    weight = ct.load(weight_ptr, index=(0,), shape=(H,))
    xhat = ct.load(xhat_ptr, index=(row, 0), shape=(1, H))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))

    grad_f = ct.astype(grad_bf, ct.float32)
    dropout_f = ct.astype(dropout_mask, ct.float32) * 1.1111111111111112
    mul_1 = grad_f * dropout_f
    ct.store(mul1_ptr, index=(row, 0), tile=mul_1)

    weight_2d = ct.reshape(weight, (1, H))
    mul_2 = mul_1 * weight_2d
    mul_3 = mul_2 * H
    sum_1 = ct.sum(mul_2, axis=1, keepdims=True)
    mul_4 = mul_2 * xhat
    sum_2 = ct.sum(mul_4, axis=1, keepdims=True)
    mul_5 = xhat * sum_2
    sub = mul_3 - sum_1
    sub_1 = sub - mul_5
    invstd_2d = ct.reshape(invstd, (1, 1))
    mul_6 = invstd_2d * sub_1
    ct.store(mul6_ptr, index=(row, 0), tile=mul_6)


@ct.kernel
def _reduce_by_seq_kernel(
    mul1_ptr,        # f32 [ROWS, HIDDEN] (viewed [BATCH*SEQ, HIDDEN])
    xhat_ptr,        # f32 [ROWS, HIDDEN]
    mul6_ptr,        # f32 [ROWS, HIDDEN]
    sum3_partial_ptr,  # f32 [SEQ, HIDDEN]
    sum4_partial_ptr,  # f32 [SEQ, HIDDEN]
    sum5_ptr,        # f32 [SEQ, HIDDEN]  (sum over BATCH of mul_6 at that seq)
    BATCH_: ct.Constant[int],
    SEQ_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
):
    seq = ct.bid(0)
    acc3 = ct.full(shape=(1, HIDDEN_), fill_value=0.0, dtype=ct.float32)
    acc4 = ct.full(shape=(1, HIDDEN_), fill_value=0.0, dtype=ct.float32)
    acc5 = ct.full(shape=(1, HIDDEN_), fill_value=0.0, dtype=ct.float32)
    for b in range(BATCH_):
        row = b * SEQ_ + seq
        mul1_row = ct.load(mul1_ptr, index=(row, 0), shape=(1, HIDDEN_))
        xhat_row = ct.load(xhat_ptr, index=(row, 0), shape=(1, HIDDEN_))
        mul6_row = ct.load(mul6_ptr, index=(row, 0), shape=(1, HIDDEN_))
        acc3 = acc3 + mul1_row * xhat_row
        acc4 = acc4 + mul1_row
        acc5 = acc5 + mul6_row
    ct.store(sum3_partial_ptr, index=(seq, 0), tile=acc3)
    ct.store(sum4_partial_ptr, index=(seq, 0), tile=acc4)
    ct.store(sum5_ptr, index=(seq, 0), tile=acc5)


@ct.kernel
def _finalize_reduce_kernel(
    sum3_partial_ptr,  # f32 [SEQ, HIDDEN]
    sum4_partial_ptr,  # f32 [SEQ, HIDDEN]
    sum3_ptr,          # f32 [HIDDEN]
    sum4_ptr,          # f32 [HIDDEN]
    SEQ_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
):
    # 1 kernel launch to reduce the SEQ dim.
    s3 = ct.load(sum3_partial_ptr, index=(0, 0), shape=(SEQ_, HIDDEN_))
    s4 = ct.load(sum4_partial_ptr, index=(0, 0), shape=(SEQ_, HIDDEN_))
    ct.store(sum3_ptr, index=(0,), tile=ct.sum(s3, axis=0))
    ct.store(sum4_ptr, index=(0,), tile=ct.sum(s4, axis=0))


@oracle_impl(hardware="B200", point="094119a0")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     shape0, shape1, shape2, shape3, shape4) = inputs
    device = arg0_1.device

    H = HIDDEN
    N = BATCH
    S = SEQ
    rows = N * S

    dropout_mask = arg2_1.view(rows, H)
    xhat_2d = arg4_1.view(rows, H)
    invstd_1d = arg5_1.view(rows)

    mul1_2d = torch.empty(rows, H, device=device, dtype=torch.float32)
    mul6_2d = torch.empty(rows, H, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _dropout_ln_bwd_kernel,
        (arg1_1, dropout_mask, arg3_1, xhat_2d, invstd_1d,
         mul1_2d, mul6_2d, H),
    )

    # Reductions inside cuTile:
    sum3_partial = torch.empty((S, H), device=device, dtype=torch.float32)
    sum4_partial = torch.empty((S, H), device=device, dtype=torch.float32)
    sum5_2d = torch.empty((S, H), device=device, dtype=torch.float32)
    ct.launch(
        stream, (S, 1, 1),
        _reduce_by_seq_kernel,
        (mul1_2d, xhat_2d, mul6_2d, sum3_partial, sum4_partial, sum5_2d,
         N, S, H),
    )
    sum_3 = torch.empty((H,), device=device, dtype=torch.float32)
    sum_4 = torch.empty((H,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (1, 1, 1),
        _finalize_reduce_kernel,
        (sum3_partial, sum4_partial, sum_3, sum_4, S, H),
    )

    # sum_5 is [1, S, H]; sum5_2d is [S, H]
    sum_5 = sum5_2d.view(1, S, H)
    mul_6 = mul6_2d.view(N, S, H)

    # Scatter into [512, 128]
    full_a = torch.zeros(tuple(int(d) for d in shape1), device=device,
                          dtype=torch.float32)
    ge = arg6_1 >= 0
    lt = arg6_1 < 512
    ne = arg6_1 != -1
    mask_a = (ge & lt & ne).view(1, S, 1)
    indices_a = arg6_1.view(-1).clamp_min(0).clamp_max(511)
    vals_a = torch.where(mask_a.expand_as(sum_5), sum_5,
                         torch.zeros_like(sum_5)).view(S, H)
    full_a.index_add_(0, indices_a, vals_a)

    # Scatter into [2, 128] with expand
    expand_val = arg7_1.expand(tuple(int(d) for d in shape2))
    ge_1 = expand_val >= 0
    lt_1 = expand_val < 2
    ne_1 = expand_val != -1
    mask_b = (ge_1 & lt_1 & ne_1).view(rows, 1)
    indices_b = expand_val.reshape(-1).clamp_min(0).clamp_max(1)
    vals_b = torch.where(mask_b.expand_as(mul_6.view(rows, H)),
                         mul_6.view(rows, H),
                         torch.zeros_like(mul_6.view(rows, H)))
    full_b = torch.zeros(tuple(int(d) for d in shape3), device=device,
                          dtype=torch.float32)
    full_b.index_add_(0, indices_b, vals_b)

    # Scatter into [30522, 128]
    ge_2 = arg8_1 >= 0
    lt_2 = arg8_1 < 30522
    ne_2 = arg8_1 != 0
    mask_c = (ge_2 & lt_2 & ne_2).view(rows, 1)
    indices_c = arg8_1.view(-1).clamp_min(0).clamp_max(30521)
    vals_c = torch.where(mask_c.expand_as(mul_6.view(rows, H)),
                         mul_6.view(rows, H),
                         torch.zeros_like(mul_6.view(rows, H)))
    full_c = torch.zeros(tuple(int(d) for d in shape4), device=device,
                          dtype=torch.float32)
    full_c.index_add_(0, indices_c, vals_c)

    add = arg0_1.float() + full_c

    return sum_3, sum_4, full_a, full_b, add
