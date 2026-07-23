"""cuTile port of sum_sum_sum_4aae5698dd79: MT5 attention-backward with bucket scatter.

Returns 4 outputs (view_4, bucket, view_9, bucket_1).
Strategy: torch for the elementwise softmax-backward chain; cuTile kernel for
the batch-dim sum reduction (sum_2 = sum(add_8, dim=0)) that feeds the scatter.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK = 128


@ct.kernel
def _sum_dim0_kernel(
    x_ptr,     # f32 [B, D]
    out_ptr,   # f32 [D]
    B_: ct.Constant[int],
    D_: ct.Constant[int],
):
    """Sum along dim=0 of an [B, D] tensor. One program handles a column tile."""
    col_tile = ct.bid(0)
    acc = ct.zeros((BLOCK,), dtype=ct.float32)
    for row in range(B_):
        x = ct.load(x_ptr, index=(row, col_tile), shape=(1, BLOCK))
        x_1d = ct.reshape(x, (BLOCK,))
        acc = acc + x_1d
    ct.store(out_ptr, index=(col_tile,), tile=acc)


def _do_branch(
    residuals, arg_bmm, arg_keep, arg_order, arg_fill, arg_logits, arg_bias,
    arg_row_shift, arg_denom, arg_index, use_order_mask, device,
):
    # Sum residuals in bf16→f32 sequential add (matches eager convert+add chain)
    add = residuals[0].to(torch.float32)
    for r in residuals[1:]:
        add = add + r.to(torch.float32)

    # Dropout keep scaling
    ct7 = arg_keep.to(torch.bfloat16)
    mul_drop = ct7 * 1.1111111111111112
    mul_1 = arg_bmm * mul_drop
    ct8 = mul_1.to(torch.float32)

    # Bias / mask
    if use_order_mask:
        u = arg_order.unsqueeze(3)
        u1 = arg_order.unsqueeze(2)
        le = u1 <= u
        expand = le.expand(32, 1, 128, 128)
        neg_inf = torch.full((), -3.4028234663852886e38, device=device, dtype=torch.float32)
        where = torch.where(expand, arg_fill, neg_inf)
    else:
        where = torch.zeros((32, 1, 128, 128), device=device, dtype=torch.float32)

    perm = arg_bias.permute(2, 0, 1).unsqueeze(0)
    add_6 = perm + where
    add_7 = arg_logits + add_6
    ct9 = add_7.to(torch.bfloat16)
    ct10 = ct9.to(torch.float32)
    sub = ct10 - arg_row_shift
    exp = torch.exp(sub)
    div = exp / arg_denom
    mul_2 = ct8 * div
    sum_1 = mul_2.sum(dim=-1, keepdim=True)
    neg = -div
    fma = torch.addcmul(mul_2, neg, sum_1)
    ct11 = fma.to(torch.bfloat16)
    view_2 = ct11.view(192, 128, 128)

    # Bucket accumulate: sum(add + ct11.f32, dim=0) → [6, 128, 128] → permute [128, 128, 6]
    add_8 = add + ct11.to(torch.float32)

    # cuTile kernel to reduce dim 0 over add_8: [32, 6*128*128] → [6*128*128]
    B = 32
    D = 6 * 128 * 128  # 98304
    x_2d = add_8.reshape(B, D).contiguous()
    out_flat = torch.empty((D,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    num_col_tiles = (D + BLOCK - 1) // BLOCK
    ct.launch(stream, (num_col_tiles, 1, 1), _sum_dim0_kernel,
              (x_2d, out_flat, B, D))
    sum_2 = out_flat.view(6, 128, 128)
    permute_1 = sum_2.permute(1, 2, 0).contiguous()  # [128, 128, 6]

    full_1 = torch.ones((128, 128, 1), device=device, dtype=torch.bool)
    full_2 = torch.zeros((32, 6), device=device, dtype=torch.float32)
    bucket = torch.ops.aten._unsafe_masked_index_put_accumulate(
        full_2, full_1, [arg_index], permute_1
    )
    return view_2, bucket


@oracle_impl(hardware="B200", point="2f93d7aa")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
        arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1,
        arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1,
        *_shape_params,
    ) = inputs
    device = arg0_1.device

    view_bmm_1 = arg7_1.view(32, 6, 128, 128)
    view_bmm_2 = arg23_1.view(32, 6, 128, 128)
    view_logits_1 = arg11_1.view(32, 6, 128, 128)
    view_logits_2 = arg25_1.view(32, 6, 128, 128)

    view_2_1, bucket_1 = _do_branch(
        (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1),
        view_bmm_1, arg8_1, arg9_1, arg10_1, view_logits_1, arg12_1,
        arg13_1, arg14_1, arg15_1,
        use_order_mask=True, device=device,
    )
    view_2_2, bucket_2 = _do_branch(
        (arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1),
        view_bmm_2, arg24_1, arg9_1, arg10_1, view_logits_2, arg26_1,
        arg27_1, arg28_1, arg29_1,
        use_order_mask=False, device=device,
    )
    return view_2_1, bucket_1, view_2_2, bucket_2
