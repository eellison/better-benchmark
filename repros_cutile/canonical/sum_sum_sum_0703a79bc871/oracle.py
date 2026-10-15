"""cuTile port of sum_sum_sum_0703a79bc871: MobileViT LN-backward + patch fold."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _ln_backward_kernel(
    grad_ptr,        # bf16 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    normed_bf_ptr,   # bf16 [rows, HIDDEN]
    mean_ptr,        # f32 [rows]
    invstd_ptr,      # f32 [rows]
    grad_out_ptr,    # bf16 padded [rows, BLOCK_H]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    grad = ct.load(grad_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    normed_bf = ct.load(normed_bf_ptr, index=(row, 0), shape=(1, BLOCK_H),
                        padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(row,), shape=(1,))

    grad_f = ct.astype(grad, ct.float32)
    normed_f = ct.astype(normed_bf, ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    mul = grad_f * weight_2d
    # NOTE: repro hardcodes multiplier 144 (from first shape), reused verbatim
    # for all shape points -> match that literal.
    mul_hidden = mul * 144.0

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    mul_masked = ct.where(col_mask, mul, 0.0)
    sum_1 = ct.sum(mul_masked)

    sub = normed_f - mean
    mul_2 = sub * invstd
    mul_3 = mul * mul_2
    mul_3_masked = ct.where(col_mask, mul_3, 0.0)
    sum_2 = ct.sum(mul_3_masked)

    mul_4 = mul_2 * sum_2
    sub_1 = mul_hidden - sum_1
    sub_2 = sub_1 - mul_4
    div_val = invstd / 144.0
    mul_5 = div_val * sub_2
    ct.store(grad_out_ptr, index=(row, 0), tile=ct.astype(mul_5, ct.bfloat16))


@oracle_impl(hardware="B200", point="30b03cad", ROWS=131072, CHANNELS=144, PATCHES=256,
             OUT_H=32, OUT_W=32, BLOCK_M=32, BLOCK_C=256, FINAL_BLOCK_C=8,
             LAYOUT_ROW_BLOCK=256, LAYOUT_C_BLOCK=4, SPLIT_REDUCE=True,
             REDUCE_BLOCK_R=256, REDUCE_BLOCK_C=16)
@oracle_impl(hardware="B200", point="1c6da2dd", ROWS=32768, CHANNELS=192, PATCHES=64,
             OUT_H=16, OUT_W=16, BLOCK_M=64, BLOCK_C=256, FINAL_BLOCK_C=8,
             LAYOUT_ROW_BLOCK=256, LAYOUT_C_BLOCK=4, SPLIT_REDUCE=False,
             REDUCE_BLOCK_R=256, REDUCE_BLOCK_C=16)
@oracle_impl(hardware="B200", point="0c9dc299", ROWS=8192, CHANNELS=240, PATCHES=16,
             OUT_H=8, OUT_W=8, BLOCK_M=64, BLOCK_C=256, FINAL_BLOCK_C=8,
             LAYOUT_ROW_BLOCK=256, LAYOUT_C_BLOCK=4, SPLIT_REDUCE=False,
             REDUCE_BLOCK_R=256, REDUCE_BLOCK_C=16)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        shape0, shape1, shape2, shape3,
    ) = inputs
    device = arg0_1.device

    hidden = int(arg1_1.shape[0])
    total = int(arg0_1.numel())
    rows = total // hidden
    # Round up to next power of 2 (144 -> 256, 192 -> 256, 240 -> 256)
    BLOCK_H = 1
    while BLOCK_H < hidden:
        BLOCK_H <<= 1

    grad_bf = arg0_1.view(rows, hidden)
    normed_bf = arg2_1.view(rows, hidden)
    mean_1d = arg3_1.view(rows)
    invstd_1d = arg4_1.view(rows)

    grad_out_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _ln_backward_kernel,
        (grad_bf, arg1_1, normed_bf, mean_1d, invstd_1d, grad_out_pad,
         hidden, BLOCK_H),
    )
    grad_out_bf = grad_out_pad.narrow(1, 0, hidden).contiguous().view(*[int(d) for d in shape0])

    grad_f = arg0_1.view(rows, hidden).to(torch.float32)
    normed_f = arg2_1.view(rows, hidden).to(torch.float32)
    sub = normed_f - arg3_1.view(rows, 1)
    mul_2 = sub * arg4_1.view(rows, 1)
    mul_6 = grad_f * mul_2
    sum_3 = mul_6.sum(dim=0)
    sum_4 = grad_f.sum(dim=0)

    add = arg5_1 + grad_out_bf
    view_1 = add.view(*[int(d) for d in shape1])
    permute = view_1.permute(0, 3, 2, 1)
    clone = permute.contiguous()
    view_2 = clone.view(*[int(d) for d in shape2])
    permute_1 = view_2.permute(0, 2, 1, 3)
    clone_1 = permute_1.contiguous()
    view_3 = clone_1.view(*[int(d) for d in shape3])

    return sum_3, sum_4, view_3
