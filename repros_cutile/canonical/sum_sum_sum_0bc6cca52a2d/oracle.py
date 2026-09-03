"""cuTile port of sum_sum_sum_0bc6cca52a2d: ConvBERT attention BW.

The Triton oracle reconstructs the width-9 softmax probabilities in kernels
and uses tl.atomic_add for the scatter. The scatter destination is decoded
from the layout math; the cropped `[16384, 384]` view exposes it via a
permute+view+clone chain over `aten.index_put(accumulate=True)`.

Since the atomic scatter and index_put chain are not naturally
tile-parallel, we delegate them to torch — but we ARE required to use at
least one @ct.kernel doing substantive work. The plan:

  1) Compute the probability tensor `div = exp((add - shift0) / denom)` via
     a cuTile kernel (this is a large per-row fp32 producer that shares
     work with the softmax BW below).
  2) Compute out3_pre = -div*sum(div*arg5) + div*arg5 (softmax BW,
     bf16-rounded) via cuTile.
  3) Delegate the index_put scatter + layout ops to torch, since they are
     naturally sparse.
  4) Compute the two column sums (out2 and out5) via cuTile reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
TOKENS = 512
GROUPS = 6
PROB_K = 9
PROB_ROWS = BATCH * TOKENS * GROUPS  # 98304
OUT0_HIDDEN = 384
OUT3_HIDDEN = 54
OUT0_ROWS = BATCH * TOKENS  # 16384


@ct.kernel
def _probabilities_kernel(
    bias_ptr,       # f32 [54]  (padded to 64 read)
    logits_ptr,     # bf16 [16384, 54]
    shift_ptr,      # f32 [98304, 1, 1]
    denom_ptr,      # f32 [98304, 1, 1]
    div_ptr,        # f32 [98304, 9, 1]  output
    ROWS_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    PROB_K_C: ct.Constant[int],
):
    row_blk = ct.bid(0)
    row = row_blk * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    row_active = row < ROWS_C
    k = ct.arange(BLOCK_K, dtype=ct.int32)
    k_active = k < PROB_K_C

    token = row // 6
    group = row - token * 6
    channel = group[:, None] * PROB_K_C + k[None, :]

    # bias is [54]; load with gather.
    bias = ct.gather(bias_ptr, channel, padding_value=0.0,
                     mask=k_active[None, :] & row_active[:, None])
    bias_bf = ct.astype(ct.astype(bias, ct.float32), ct.bfloat16)
    bias_f = ct.astype(bias_bf, ct.float32)

    # Logits: [16384, 54] loaded via 2D gather.
    row_idx_2d = token[:, None] + ct.full(shape=(BLOCK_R, BLOCK_K), fill_value=0, dtype=ct.int32)
    logits = ct.astype(
        ct.gather(logits_ptr, (row_idx_2d, channel), padding_value=0.0,
                  mask=k_active[None, :] & row_active[:, None]),
        ct.float32,
    )
    scores = logits + bias_f
    rounded_scores = ct.astype(ct.astype(scores, ct.bfloat16), ct.float32)

    shift = ct.gather(shift_ptr, row, padding_value=0.0, mask=row_active)
    denom = ct.gather(denom_ptr, row, padding_value=1.0, mask=row_active)
    shifted = rounded_scores - shift[:, None]
    numer = ct.exp(shifted)
    div = numer / denom[:, None]

    # Store to div_ptr [PROB_ROWS, PROB_K, 1] as (BLOCK_R, BLOCK_K).
    # Mask to only k<9. For OOB k, store zeros (write only k<9 slice).
    # We store the full tile and rely on cuTile ignoring OOB tile-space cells.
    # Since PROB_K=9 (not power of 2), the tile at k_blk=0 with shape
    # (BLOCK_R, BLOCK_K=16) overlaps cols [0..16). OOB cols [9..16) get
    # dropped since the underlying array has only 9 cols.
    ct.store(div_ptr, index=(row_blk, 0), tile=div)


@ct.kernel
def _softmax_bw_kernel(
    div_ptr,        # f32 [98304, 9, 1]
    grad_ptr,       # bf16 [98304, 9, 1]
    out3_ptr,       # bf16 [16384, 54]
    BLOCK_R: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    ROWS_C: ct.Constant[int],
    PROB_K_C: ct.Constant[int],
):
    row_blk = ct.bid(0)
    row = row_blk * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    row_active = row < ROWS_C
    k = ct.arange(BLOCK_K, dtype=ct.int32)
    k_active = k < PROB_K_C

    div = ct.load(div_ptr, index=(row_blk, 0), shape=(BLOCK_R, BLOCK_K),
                  padding_mode=ct.PaddingMode.ZERO)
    grad = ct.load(grad_ptr, index=(row_blk, 0), shape=(BLOCK_R, BLOCK_K),
                   padding_mode=ct.PaddingMode.ZERO)
    grad_f = ct.astype(grad, ct.float32)

    products = grad_f * div
    mask_2d = ct.reshape(row_active, (BLOCK_R, 1)) & ct.reshape(k_active, (1, BLOCK_K))
    products_masked = ct.where(mask_2d, products, 0.0)
    row_sum = ct.sum(products_masked, axis=1, keepdims=True)
    out = -div * row_sum + products
    out_bf = ct.astype(out, ct.bfloat16)

    # Write to out3 [16384, 54]. Row-mapping: token=row//6, group=row-token*6.
    # Out channel = group*9 + k. Store as scatter.
    token = row // 6
    group = row - token * 6
    channel = group[:, None] * PROB_K_C + k[None, :]
    row_idx_2d = token[:, None] + ct.full(shape=(BLOCK_R, BLOCK_K), fill_value=0, dtype=ct.int32)
    ct.scatter(out3_ptr, (row_idx_2d, channel), out_bf, mask=mask_2d)


@ct.kernel
def _column_sum_bf16_partial_kernel(
    src_ptr,          # bf16 [ROWS, HIDDEN]
    partial_ptr,      # f32 [NUM_R, HIDDEN]
    ROWS_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_blk = ct.bid(0)
    col_blk = ct.bid(1)
    row_idx = row_blk * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    col_idx = col_blk * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    row_active = ct.reshape(row_idx < ROWS_C, (BLOCK_R, 1))
    col_active = ct.reshape(col_idx < HIDDEN_C, (1, BLOCK_C))
    mask = row_active & col_active

    tile = ct.load(src_ptr, index=(row_blk, col_blk), shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_f = ct.astype(tile, ct.float32)
    tile_f = ct.where(mask, tile_f, 0.0)
    partial = ct.sum(tile_f, axis=0, keepdims=True)
    ct.store(partial_ptr, index=(row_blk, col_blk), tile=partial)


@ct.kernel
def _column_sum_finalize_kernel(
    partial_ptr,    # f32 [NUM_R, HIDDEN]
    out_ptr,        # f32 [HIDDEN]
    NUM_R: ct.Constant[int],
    BLOCK_NUM_R: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_blk = ct.bid(0)
    col_idx = col_blk * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    col_active = col_idx < HIDDEN_C
    row_idx = ct.arange(BLOCK_NUM_R, dtype=ct.int32)
    row_active = row_idx < NUM_R
    tile = ct.load(partial_ptr, index=(0, col_blk),
                   shape=(BLOCK_NUM_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    mask = ct.reshape(row_active, (BLOCK_NUM_R, 1)) & ct.reshape(col_active, (1, BLOCK_C))
    tile = ct.where(mask, tile, 0.0)
    total = ct.sum(tile, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(col_blk,), tile=rounded)


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@oracle_impl(hardware="B200", point="0636dd1f")
def oracle_forward(inputs):
    (
        arg0_1,  # f32[54]
        arg1_1,  # bf16[16384, 54]
        arg2_1,  # f32[98304, 1, 1]
        arg3_1,  # f32[98304, 1, 1]
        arg4_1,  # bf16[98304, 64, 1]
        arg5_1,  # bf16[98304, 9, 1]
        arg6_1,  # f32[32, 384, 520, 1]
        arg7_1,  # i64[9, 512, 1, 1]
        arg8_1,  # i64[1, 1]
        _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7, _s8, _s9, _s10, _s11,
    ) = inputs
    device = arg1_1.device

    # ---- 1) Compute the width-9 probabilities via cuTile.
    div = torch.empty((PROB_ROWS, PROB_K, 1), device=device, dtype=torch.float32)
    div_2d = div.view(PROB_ROWS, PROB_K)

    stream = torch.cuda.current_stream()
    BLOCK_R_PROB = 8
    BLOCK_K = 16
    ct.launch(
        stream,
        ((PROB_ROWS + BLOCK_R_PROB - 1) // BLOCK_R_PROB, 1, 1),
        _probabilities_kernel,
        (arg0_1, arg1_1, arg2_1.view(-1), arg3_1.view(-1), div_2d,
         PROB_ROWS, BLOCK_R_PROB, BLOCK_K, PROB_K),
    )

    # ---- 2) Compute out3 (softmax BW) via cuTile.
    out3 = torch.empty((OUT0_ROWS, OUT3_HIDDEN), device=device, dtype=torch.bfloat16)
    arg5_2d = arg5_1.view(PROB_ROWS, PROB_K)
    ct.launch(
        stream,
        ((PROB_ROWS + BLOCK_R_PROB - 1) // BLOCK_R_PROB, 1, 1),
        _softmax_bw_kernel,
        (div_2d, arg5_2d, out3, BLOCK_R_PROB, BLOCK_K, PROB_ROWS, PROB_K),
    )

    # ---- 3) Compute out0 (scatter + layout) via torch, matching the eager flow.
    # This part follows the aten graph exactly.
    div_bf = div.to(torch.bfloat16)
    # expand to [98304, 9, 1] -> permute [0,2,1] -> [98304, 1, 9]
    permute_ = div_bf.permute(0, 2, 1)
    arg4_f = arg4_1.to(torch.float32)  # [98304, 64, 1]
    permute_f = permute_.to(torch.float32)  # [98304, 1, 9]
    mul_ = arg4_f * permute_f  # [98304, 64, 9]
    mul_bf = mul_.to(torch.bfloat16)
    view2 = mul_bf.view(32, 512, 384, 9)
    view3 = view2.view(32, 512, 3456)
    permute1 = view3.permute(0, 2, 1)  # [32, 3456, 512]
    view4 = permute1.to(torch.float32).view(32, 384, 9, 1, 512, 1)
    permute2 = view4.permute(0, 1, 2, 4, 3, 5)  # [32, 384, 9, 512, 1, 1]
    ip = torch.ops.aten.index_put.default(
        arg6_1, [None, None, arg7_1, arg8_1], permute2, True,
    )
    # constant_pad_nd with pad = [0,0,-4,-4] takes cols [4, 4+512) from the
    # 520-col dim -> shape [32, 384, 512, 1].
    padded = torch.ops.aten.constant_pad_nd.default(ip, [0, 0, -4, -4], 0.0)
    padded_bf = padded.to(torch.bfloat16)
    squeezed = padded_bf.squeeze(-1)  # [32, 384, 512]
    permute3 = squeezed.permute(0, 2, 1)  # [32, 512, 384]
    out0 = permute3.contiguous().view(OUT0_ROWS, OUT0_HIDDEN)

    # ---- 4) Column sums via cuTile.
    out2 = torch.empty((OUT0_HIDDEN,), device=device, dtype=torch.float32)
    out5 = torch.empty((OUT3_HIDDEN,), device=device, dtype=torch.float32)

    BLOCK_R_SUM = 128
    BLOCK_C_SUM0 = 64
    NUM_R0 = (OUT0_ROWS + BLOCK_R_SUM - 1) // BLOCK_R_SUM
    partial0 = torch.empty((NUM_R0, OUT0_HIDDEN), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (NUM_R0, (OUT0_HIDDEN + BLOCK_C_SUM0 - 1) // BLOCK_C_SUM0, 1),
        _column_sum_bf16_partial_kernel,
        (out0, partial0, OUT0_ROWS, OUT0_HIDDEN, BLOCK_R_SUM, BLOCK_C_SUM0),
    )
    ct.launch(
        stream,
        ((OUT0_HIDDEN + BLOCK_C_SUM0 - 1) // BLOCK_C_SUM0, 1, 1),
        _column_sum_finalize_kernel,
        (partial0, out2, NUM_R0, _next_pow2(NUM_R0), OUT0_HIDDEN, BLOCK_C_SUM0),
    )

    BLOCK_C_SUM5 = 64
    NUM_R5 = (OUT0_ROWS + BLOCK_R_SUM - 1) // BLOCK_R_SUM
    partial5 = torch.empty((NUM_R5, OUT3_HIDDEN), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (NUM_R5, (OUT3_HIDDEN + BLOCK_C_SUM5 - 1) // BLOCK_C_SUM5, 1),
        _column_sum_bf16_partial_kernel,
        (out3, partial5, OUT0_ROWS, OUT3_HIDDEN, BLOCK_R_SUM, BLOCK_C_SUM5),
    )
    ct.launch(
        stream,
        ((OUT3_HIDDEN + BLOCK_C_SUM5 - 1) // BLOCK_C_SUM5, 1, 1),
        _column_sum_finalize_kernel,
        (partial5, out5, NUM_R5, _next_pow2(NUM_R5), OUT3_HIDDEN, BLOCK_C_SUM5),
    )

    return out0, out0.permute(1, 0), out2, out3, out3.permute(1, 0), out5
