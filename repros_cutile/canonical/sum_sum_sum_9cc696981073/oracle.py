"""cuTile port of sum_sum_sum_9cc696981073: DeiT LayerNorm-backward + patch fold.

Moves the cooperative-split-K per-channel reductions (sum_3, sum_4, sum_8) into
cuTile kernels, matching Triton's partial + finalize structure. The atomic-add
token-sum reductions (sum_5, sum_6, sum_7) stay in torch because cuTile lacks
masked atomic scatter.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768


@ct.kernel
def _ln_backward_kernel(
    grad_ptr,        # bf16 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    add_ptr,         # f32 [rows, HIDDEN]
    mean_ptr,        # f32 [rows]
    invstd_ptr,      # f32 [rows]
    grad_input_ptr,  # f32 padded [rows, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    grad = ct.load(grad_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    add_val = ct.load(add_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))

    grad_f = ct.astype(grad, ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    mul = grad_f * weight_2d
    mul_hidden = mul * HIDDEN_

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    mul_masked = ct.where(col_mask, mul, 0.0)
    sum_1 = ct.sum(mul_masked)

    sub = add_val - mean
    mul_2 = sub * invstd
    mul_3 = mul * mul_2
    mul_3_masked = ct.where(col_mask, mul_3, 0.0)
    sum_2 = ct.sum(mul_3_masked)

    mul_4 = mul_2 * sum_2
    sub_1 = mul_hidden - sum_1
    sub_2 = sub_1 - mul_4
    div_val = invstd * (1.0 / HIDDEN_)
    mul_5 = div_val * sub_2
    ct.store(grad_input_ptr, index=(row, 0), tile=mul_5)


@ct.kernel
def _channel_sum_dot_kernel(
    x_ptr,           # f32 [CHANNELS, K_TOTAL]
    centered_ptr,    # f32 [CHANNELS, K_TOTAL]
    sum_x_ptr,       # f32 [CHANNELS] (sum_4)
    sum_dot_ptr,     # f32 [CHANNELS] (sum_3)
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    sum_x_acc = ct.zeros((), dtype=ct.float32)
    sum_dot_acc = ct.zeros((), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        x_f = ct.load(
            x_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        c_f = ct.load(
            centered_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ks = ct.arange(BLOCK_K, dtype=ct.int32) + kb * BLOCK_K
        valid = ks < K_TOTAL_
        valid_2d = ct.reshape(valid, (1, BLOCK_K))
        zero_f = ct.full((1, BLOCK_K), 0.0, dtype=ct.float32)
        x_m = ct.where(valid_2d, x_f, zero_f)
        c_m = ct.where(valid_2d, c_f, zero_f)
        sum_x_acc = sum_x_acc + ct.sum(x_m)
        sum_dot_acc = sum_dot_acc + ct.sum(x_m * c_m)
    ct.store(sum_x_ptr, index=(c,), tile=ct.reshape(sum_x_acc, (1,)))
    ct.store(sum_dot_ptr, index=(c,), tile=ct.reshape(sum_dot_acc, (1,)))


@ct.kernel
def _channel_sum_bf16_kernel(
    x_ptr,               # bf16 [C, K_TOTAL]
    sum_out_ptr,         # f32 [C]
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    N_K_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    acc_f = ct.zeros((), dtype=ct.float32)
    for kb in range(N_K_BLOCKS):
        x_bf = ct.load(
            x_ptr, index=(c, kb), shape=(1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ks = ct.arange(BLOCK_K, dtype=ct.int32) + kb * BLOCK_K
        valid = ks < K_TOTAL_
        valid_2d = ct.reshape(valid, (1, BLOCK_K))
        zero_f = ct.full((1, BLOCK_K), 0.0, dtype=ct.float32)
        x_f = ct.astype(x_bf, ct.float32)
        x_masked = ct.where(valid_2d, x_f, zero_f)
        acc_f = acc_f + ct.sum(x_masked)
    # Round to bf16 then back to f32 to match Triton's `.to(bf16).to(f32)`.
    acc_bf = ct.astype(acc_f, ct.bfloat16)
    acc_f_out = ct.astype(acc_bf, ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(acc_f_out, (1,)))


@oracle_impl(hardware="B200", point="92596d7a")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        shape0, shape1,
    ) = inputs
    device = arg0_1.device
    n = 128
    seq = 198
    hidden = HIDDEN
    rows = n * seq
    BLOCK_H = 1024
    BLOCK_K = 4096

    grad_bf = arg0_1.view(n, seq, hidden)
    add_val = arg2_1 + arg3_1

    grad_input_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    grad_view = grad_bf.reshape(rows, hidden).contiguous()
    add_view = add_val.reshape(rows, hidden).contiguous()
    mean_view = arg4_1.view(rows)
    invstd_view = arg5_1.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _ln_backward_kernel,
        (grad_view, arg1_1, add_view, mean_view, invstd_view, grad_input_pad,
         hidden, BLOCK_H),
    )
    grad_input = grad_input_pad.narrow(1, 0, hidden).contiguous().view(n, seq, hidden)

    grad_f = grad_bf.to(torch.float32)
    sub = add_val - arg4_1
    mul_2 = sub * arg5_1

    # Cooperative-split-K per-channel reductions (sum_3, sum_4) via cuTile.
    # sum_4 = grad_f.sum(dim=[0,1]); sum_3 = (grad_f * mul_2).sum(dim=[0,1]).
    grad_c = grad_f.reshape(rows, hidden).t().contiguous()   # [C, N*T]
    mul_2_c = mul_2.reshape(rows, hidden).t().contiguous()   # [C, N*T]
    sum_4 = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_3 = torch.empty((hidden,), device=device, dtype=torch.float32)
    K_TOTAL = rows
    N_K_BLOCKS = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    ct.launch(
        stream, (hidden, 1, 1), _channel_sum_dot_kernel,
        (grad_c, mul_2_c, sum_4, sum_3, K_TOTAL, BLOCK_K, N_K_BLOCKS),
    )

    add_1 = arg6_1 + grad_input
    sum_5 = add_1.sum(dim=0, keepdim=True, dtype=torch.float32)

    slice_1 = add_1[:, 0:1]
    slice_2 = add_1[:, 1:2]
    slice_3 = add_1[:, 2:198]
    ce1 = slice_3.to(torch.bfloat16)

    sum_6 = slice_2.sum(dim=0, keepdim=True, dtype=torch.float32)
    sum_7 = slice_1.sum(dim=0, keepdim=True, dtype=torch.float32)

    permute = ce1.permute(0, 2, 1)
    view_1 = permute.reshape(*[int(d) for d in shape1])

    # Per-channel bf16 patch sum (sum_8) via cuTile — matches Triton's
    # `.to(bf16).to(f32)` epilogue on the accumulated result.
    view_1_c = view_1.permute(1, 0, 2, 3).contiguous().view(hidden, -1)  # [C, K2]
    K2 = view_1_c.shape[1]
    N_K_BLOCKS2 = (K2 + BLOCK_K - 1) // BLOCK_K
    sum_8_f = torch.empty((hidden,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (hidden, 1, 1), _channel_sum_bf16_kernel,
        (view_1_c, sum_8_f, K2, BLOCK_K, N_K_BLOCKS2),
    )

    return sum_3, sum_4, sum_5, sum_6, sum_7, view_1, sum_8_f
