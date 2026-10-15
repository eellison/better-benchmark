"""cuTile port of amax_sum_528a3c274a41: Longformer inference sliding-window attention.

Runs the intricate score-band assembly and mask construction in torch, then
delegates the row-softmax (with query-mask zeroing) to a cuTile kernel. Final
layout is done in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
WINDOW = 513


@ct.kernel
def _softmax_bf16_kernel(
    scores_ptr,     # bf16 [rows, 513]  (padded to BLOCK_N)
    query_mask_ptr, # b8   [rows]        (True => zero-out row)
    out_ptr,        # bf16 [rows, 513]
    W: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N),
                        padding_mode=ct.PaddingMode.ZERO)
    scores_f = ct.astype(scores_bf, ct.float32)
    # Mask out padded cols with -inf so they don't affect max/exp
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < W, (1, BLOCK_N))
    ninf = ct.full((1, BLOCK_N), -float("inf"), dtype=ct.float32)
    scores_masked = ct.where(col_mask, scores_f, ninf)
    row_max = ct.max(scores_masked)
    numer = ct.exp(scores_masked - row_max)
    numer = ct.where(col_mask, numer, 0.0)
    denom = ct.sum(numer)
    probs_f = numer / denom
    qmask = ct.load(query_mask_ptr, index=(row,), shape=(1,))
    active_row = qmask == 0
    zero_f = ct.full((1, BLOCK_N), 0.0, dtype=ct.float32)
    result_f = ct.where(active_row, probs_f, zero_f)
    result_bf = ct.astype(result_f, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=result_bf)


@oracle_impl(hardware="B200", point="79c25467", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, *sp = inputs
    device = arg1_1.device
    sp_t = [tuple(int(d) for d in s) for s in sp]

    def full_zero(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, device=device, dtype=dtype)

    unsqueeze = arg0_1.unsqueeze(2).unsqueeze(3)  # b8 [8, 1024, 1, 1]

    full_1 = full_zero(sp_t[0])
    view = arg1_1.view(*sp_t[1])
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(*sp_t[2])
    constant_pad_nd = torch.nn.functional.pad(view_1, [0, 0, 0, 1], value=0.0)
    view_2 = constant_pad_nd.view(*sp_t[3])

    scaffold = full_1.clone()
    scaffold[:, 0:-1, :, 256:] = view_2[:, :, 0:256, 0:257]
    scaffold[:, -1, :, 256:] = view_2[:, -1, 256:, 0:257]
    scaffold[:, 1:, :, 0:256] = view_2[:, :, -257:-1, 257:]
    scaffold[:, 0, 1:256, 1:256] = view_2[:, 0, 0:255, -255:]

    scores = scaffold.view(*sp_t[4]).permute(0, 2, 1, 3)  # [B, 1024, 12, 513]

    iota = torch.arange(257, device=device).unsqueeze(0)
    iota_1 = torch.arange(256, device=device).unsqueeze(1)
    le = (iota - iota_1) <= 0
    ones_bf = torch.ones((256, 257), device=device, dtype=torch.bfloat16)
    zero_bf = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
    where_mask = torch.where(le, ones_bf, zero_bf)
    rev = torch.flip(where_mask, [0])
    ninf = torch.tensor(-float("inf"), device=device, dtype=torch.bfloat16)

    corner_mask = rev.unsqueeze(0).unsqueeze(2).expand(BATCH, 256, HEADS, 257).bool()
    scores[:, 0:256, :, 0:257] = torch.where(
        corner_mask, ninf, scores[:, 0:256, :, 0:257]
    )
    rev_bot = torch.flip(rev.unsqueeze(0).unsqueeze(2), [1, 3])
    corner_mask_bot = rev_bot.expand(BATCH, 256, HEADS, 257).bool()
    scores[:, -256:, :, -257:] = torch.where(
        corner_mask_bot, ninf, scores[:, -256:, :, -257:]
    )

    # Key-mask contribution over the same band
    fill_val = torch.tensor(-3.3895313892515355e38, device=device, dtype=torch.bfloat16)
    ne_mask = (arg2_1 != 0)  # b8 [8, 1024]
    ne4 = ne_mask.unsqueeze(2).unsqueeze(3)  # [8, 1024, 1, 1]
    mask_bf = ne4.to(torch.bfloat16)
    mask_where = torch.where(ne4, fill_val, mask_bf)  # bf16 [8, 1024, 1, 1]

    permute_14 = mask_where.permute(0, 2, 1, 3)
    view_10 = permute_14.reshape(*sp_t[22])
    view_11 = view_10.reshape(*sp_t[23])
    as_strided_1 = torch.as_strided(view_11, sp_t[24], sp_t[25])
    unsqueeze_9 = as_strided_1.unsqueeze(4)
    permute_15 = unsqueeze_9.permute(0, 1, 4, 2, 3)

    ones_bf16_all = torch.ones(sp_t[17], device=device, dtype=torch.bfloat16)
    permute_12 = ones_bf16_all.permute(0, 2, 1, 3)
    view_10_ = permute_12.reshape(*sp_t[18])
    view_11_ = view_10_.reshape(*sp_t[19])
    as_strided = torch.as_strided(view_11_, sp_t[20], sp_t[21])
    unsqueeze_6 = as_strided.unsqueeze(4)
    permute_13 = unsqueeze_6.permute(0, 1, 2, 4, 3)  # (B, 3, 512, 1, 1)
    mul_mask = permute_13 * permute_15
    view_14 = mul_mask.reshape(BATCH, 3, 512, 512)
    constant_pad_nd_1 = torch.nn.functional.pad(view_14, [0, 0, 0, 1], value=0.0)
    view_15 = constant_pad_nd_1.view(BATCH, 3, 512, 513)

    full_5 = full_zero((BATCH, 4, 256, 513))
    full_5[:, 0:-1, :, 256:] = view_15[:, :, 0:256, 0:257]
    full_5[:, -1, :, 256:] = view_15[:, -1, 256:, 0:257]
    full_5[:, 1:, :, 0:256] = view_15[:, :, -257:-1, 257:]
    full_5[:, 0, 1:256, 1:256] = view_15[:, 0, 0:255, -255:]

    key_mask_scores = full_5.view(BATCH, 1, 1024, 513).permute(0, 2, 1, 3)
    key_mask_scores_tl = key_mask_scores[:, 0:256, :, 0:257].clone()
    key_mask_scores[:, 0:256, :, 0:257] = torch.where(
        corner_mask[:, :, :1, :], ninf, key_mask_scores_tl
    )
    key_mask_scores_br = key_mask_scores[:, -256:, :, -257:].clone()
    key_mask_scores[:, -256:, :, -257:] = torch.where(
        corner_mask_bot[:, :, :1, :], ninf, key_mask_scores_br
    )

    # Broadcast key_mask_scores (B, 1024, 1, 513) + scores (B, 1024, 12, 513)
    scores_total = scores + key_mask_scores

    # Softmax per (B, Q, H) row with q_mask zero-out
    scores_flat = scores_total.contiguous().view(-1, WINDOW)
    rows = scores_flat.shape[0]
    q_mask_full = arg0_1.unsqueeze(2).expand(BATCH, SEQ, HEADS).contiguous().view(-1)
    q_mask_i8 = q_mask_full.to(torch.int32)

    out_flat = torch.empty_like(scores_flat)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _softmax_bf16_kernel,
        (scores_flat, q_mask_i8, out_flat, WINDOW, BLOCK_N),
    )

    out_shaped = out_flat.view(BATCH, SEQ, HEADS, WINDOW).permute(0, 2, 1, 3).contiguous()
    view_23 = out_shaped.reshape(BATCH * HEADS, 4, 256, WINDOW)
    constant_pad_nd_2 = torch.nn.functional.pad(view_23, sp_t[41], value=0.0)
    view_24 = constant_pad_nd_2.view(*sp_t[42])
    slice_57 = view_24[:, :, 0:-256]
    view_25 = slice_57.view(*sp_t[43])
    slice_58 = view_25[:, :, :, 0:-1]
    unsqueeze_14 = slice_58.unsqueeze(4)
    view_26 = unsqueeze_14.view(*sp_t[44])
    return view_26
