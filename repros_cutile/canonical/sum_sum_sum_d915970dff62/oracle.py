"""cuTile port of sum_sum_sum_d915970dff62: ConvBERT attention-gradient
scatter-reduce with softmax-backward and dependent column sums.

Structure:
  * `_zero_full_and_temp_kernel` — zero the returned full[32,384,520,1] f32
    tensor and the temp[16384, 384] f32 scratch used for scatter-add.
  * `_scatter_probabilities_kernel` — for each (row, c_block), reconstruct the
    9-way softmax probabilities via exp/(shift/denom), multiply against
    arg4[row, c], and atomic-add the bf16-rounded products into temp[out_row, out_channel]
    guarded by (idx >= 4 & idx < 516).
  * `_store_out3_kernel` — for each block of PROB_ROWS rows, reconstruct probs,
    compute softmax-backward `out = -p*sum + p*dy` in f32 then round to bf16.
  * `_finalize_out0_partials_kernel` — round temp[16384,384] to bf16 and produce
    per-row-group column partials (f32).
  * `_reduce_bf16_columns_kernel` — column sum over out3.
  * `_finalize_bf16_sum_kernel` — sum column partials and bf16-round.

Inline PTX (add.rn/mul.rn/fma.rn f32) is just RTNE — use +/*/etc. directly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
TOKENS = 512
GROUPS = 6
PROB_K = 9
PROB_ROWS = 98304
OUT0_HIDDEN = 384
OUT3_HIDDEN = 54
OUT0_ROWS = BATCH * TOKENS
PADDED_TOKENS = 520
CROP = 4
ARG4_CHANNELS = 64
FULL_NUMEL = BATCH * OUT0_HIDDEN * PADDED_TOKENS
TEMP_NUMEL = OUT0_ROWS * OUT0_HIDDEN


@ct.kernel
def _zero_full_and_temp_kernel(
    full_ptr,   # f32 [FULL_NUMEL]
    temp_ptr,   # f32 [TEMP_NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    zeros = ct.zeros((BLOCK,), dtype=ct.float32)
    m_full = offs < FULL_NUMEL
    m_temp = offs < TEMP_NUMEL
    ct.scatter(full_ptr, offs, zeros, mask=m_full)
    ct.scatter(temp_ptr, offs, zeros, mask=m_temp)


@ct.kernel
def _scatter_probabilities_kernel(
    bias_ptr,     # f32 [54]
    logits_ptr,   # bf16 [16384, 54]
    shift_ptr,    # f32 [98304]
    denom_ptr,    # f32 [98304]
    arg4_ptr,     # bf16 [98304, 64]
    index_ptr,    # i64 [9, 512]
    temp_ptr,    # f32 [16384, 384]
    BLOCK_C: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)  # 0..PROB_ROWS-1
    c_block = ct.bid(1)
    cs = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C  # (BLOCK_C,)
    ks = ct.arange(BLOCK_K, dtype=ct.int32)  # (BLOCK_K,)
    k_valid = ks < PROB_K

    # Compute per-k probs (bf16 -> f32 -> exp -> divide by denom)
    token = row // GROUPS
    group = row - token * GROUPS
    channel = group * PROB_K + ks  # (BLOCK_K,)
    row_i32 = ct.full((BLOCK_K,), row, dtype=ct.int32)
    token_i32 = ct.full((BLOCK_K,), token, dtype=ct.int32)
    bias_bf = ct.astype(
        ct.gather(bias_ptr, channel, mask=k_valid, padding_value=0), ct.float32,
    )
    bias_rt = ct.astype(ct.astype(bias_bf, ct.bfloat16), ct.float32)
    logits_bf = ct.gather(logits_ptr, (token_i32, channel), mask=k_valid, padding_value=0)
    logits_f = ct.astype(logits_bf, ct.float32)
    rounded = ct.astype(ct.astype(logits_f + bias_rt, ct.bfloat16), ct.float32)
    shift_s = ct.astype(
        ct.gather(shift_ptr, ct.full((1,), row, dtype=ct.int32), padding_value=0),
        ct.float32,
    )
    denom_s = ct.astype(
        ct.gather(denom_ptr, ct.full((1,), row, dtype=ct.int32), padding_value=1.0),
        ct.float32,
    )
    shift_bcast = ct.reshape(shift_s, (1,)) + ct.zeros((BLOCK_K,), dtype=ct.float32)
    denom_bcast = ct.reshape(denom_s, (1,)) + ct.zeros((BLOCK_K,), dtype=ct.float32)
    shifted = rounded - shift_bcast
    numer = ct.exp(shifted)
    probs = numer / denom_bcast   # (BLOCK_K,) f32

    # Compute output row (idx) for each k. index has shape (9, 512).
    seq = token - (token // TOKENS) * TOKENS
    batch = token // TOKENS
    seq_i32 = ct.full((BLOCK_K,), seq, dtype=ct.int32)
    idx_i64 = ct.gather(index_ptr, (ks, seq_i32), mask=k_valid, padding_value=-1)
    idx_i32 = ct.astype(idx_i64, ct.int32)
    live = k_valid & (idx_i32 >= 4) & (idx_i32 < PADDED_TOKENS - 4)
    out_row = ct.astype(batch, ct.int32) * TOKENS + idx_i32 - CROP  # (BLOCK_K,)

    # arg4[row, cs] bf16 -> f32
    row_bc_c = ct.full((BLOCK_C,), row, dtype=ct.int32)
    c_valid = cs < ARG4_CHANNELS
    arg4_bf = ct.gather(arg4_ptr, (row_bc_c, cs), mask=c_valid, padding_value=0)
    arg4_f = ct.astype(arg4_bf, ct.float32)  # (BLOCK_C,)

    # Compute values (BLOCK_C, BLOCK_K) = arg4[c] * probs[k] rounded to bf16
    probs_bf16_f = ct.astype(ct.astype(probs, ct.bfloat16), ct.float32)  # (BLOCK_K,)
    arg4_2d = ct.reshape(arg4_f, (BLOCK_C, 1)) + ct.zeros((BLOCK_C, BLOCK_K), dtype=ct.float32)
    probs_2d = ct.reshape(probs_bf16_f, (1, BLOCK_K)) + ct.zeros((BLOCK_C, BLOCK_K), dtype=ct.float32)
    values = ct.astype(ct.astype(arg4_2d * probs_2d, ct.bfloat16), ct.float32)  # (BLOCK_C, BLOCK_K)

    # out_channel = group * 64 + c   (broadcast to BLOCK_C, BLOCK_K)
    out_ch = ct.astype(group, ct.int32) * ARG4_CHANNELS + cs   # (BLOCK_C,)
    out_ch_2d = ct.reshape(out_ch, (BLOCK_C, 1)) + ct.zeros((BLOCK_C, BLOCK_K), dtype=ct.int32)
    out_row_2d = ct.reshape(out_row, (1, BLOCK_K)) + ct.zeros((BLOCK_C, BLOCK_K), dtype=ct.int32)
    live_2d = ct.reshape(live, (1, BLOCK_K)) & ct.reshape(c_valid, (BLOCK_C, 1))
    # temp is [16384, 384]. atomic_add lacks mask, so gate values via ct.where
    # and clamp indices to safe row (0) when masked out to satisfy check_bounds.
    zeros2d = ct.zeros((BLOCK_C, BLOCK_K), dtype=ct.float32)
    values_gated = ct.where(live_2d, values, zeros2d)
    zero_i = ct.zeros((BLOCK_C, BLOCK_K), dtype=ct.int32)
    safe_row = ct.where(live_2d, out_row_2d, zero_i)
    ct.atomic_add(temp_ptr, (safe_row, out_ch_2d), values_gated,
                  memory_order=ct.MemoryOrder.RELAXED)


@ct.kernel
def _store_out3_kernel(
    bias_ptr,     # f32 [54]
    logits_ptr,   # bf16 [16384, 54]
    shift_ptr,    # f32 [98304]
    denom_ptr,    # f32 [98304]
    arg5_ptr,     # bf16 [98304, 9]
    out3_ptr,     # bf16 [16384, 54]
    BLOCK_R: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row_block = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32) + row_block * BLOCK_R  # (BLOCK_R,)
    ks = ct.arange(BLOCK_K, dtype=ct.int32)  # (BLOCK_K,)
    row_valid = rows < PROB_ROWS
    k_valid = ks < PROB_K
    mask2 = ct.reshape(row_valid, (BLOCK_R, 1)) & ct.reshape(k_valid, (1, BLOCK_K))

    # Compute probs (BLOCK_R, BLOCK_K)
    token = rows // GROUPS  # (BLOCK_R,)
    group = rows - token * GROUPS  # (BLOCK_R,)
    channel = ct.reshape(group, (BLOCK_R, 1)) * PROB_K + ct.reshape(ks, (1, BLOCK_K))  # (BLOCK_R, BLOCK_K)
    token_2d = ct.reshape(token, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.int32)
    bias_bf = ct.astype(
        ct.gather(bias_ptr, channel, mask=mask2, padding_value=0), ct.float32,
    )
    bias_rt = ct.astype(ct.astype(bias_bf, ct.bfloat16), ct.float32)
    logits_bf = ct.gather(logits_ptr, (token_2d, channel), mask=mask2, padding_value=0)
    logits_f = ct.astype(logits_bf, ct.float32)
    rounded = ct.astype(ct.astype(logits_f + bias_rt, ct.bfloat16), ct.float32)
    shift_r = ct.astype(
        ct.gather(shift_ptr, rows, mask=row_valid, padding_value=0), ct.float32,
    )
    denom_r = ct.astype(
        ct.gather(denom_ptr, rows, mask=row_valid, padding_value=1.0), ct.float32,
    )
    shift_2d = ct.reshape(shift_r, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.float32)
    denom_2d = ct.reshape(denom_r, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.float32)
    shifted = rounded - shift_2d
    numer = ct.exp(shifted)
    probs = numer / denom_2d  # (BLOCK_R, BLOCK_K)

    # arg5[row, k]
    arg5_bf = ct.gather(arg5_ptr, (
        ct.reshape(rows, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.int32),
        ct.reshape(ks, (1, BLOCK_K)) + ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.int32),
    ), mask=mask2, padding_value=0)
    dy = ct.astype(arg5_bf, ct.float32)
    products = dy * probs
    zeros2 = ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.float32)
    products_masked = ct.where(mask2, products, zeros2)
    row_sum = ct.sum(products_masked, axis=1)  # (BLOCK_R,)
    row_sum_2d = ct.reshape(row_sum, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_K), dtype=ct.float32)
    # out = -p*row_sum + products  (FMA)
    out = -probs * row_sum_2d + products
    out_bf = ct.astype(out, ct.bfloat16)

    # Store into out3[token, channel]
    ct.scatter(out3_ptr, (token_2d, channel), out_bf, mask=mask2)


@ct.kernel
def _finalize_out0_partials_kernel(
    temp_ptr,        # f32 [16384, 384]
    out0_ptr,        # bf16 [16384, 384]
    partials_ptr,    # f32 [NUM_ROW_GROUPS, 384]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    col_group = ct.bid(1)
    rows = ct.arange(BLOCK_R, dtype=ct.int32) + row_group * BLOCK_R
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + col_group * BLOCK_C
    row_valid = rows < OUT0_ROWS
    col_valid = cols < OUT0_HIDDEN
    mask2 = ct.reshape(row_valid, (BLOCK_R, 1)) & ct.reshape(col_valid, (1, BLOCK_C))
    row_tile = ct.reshape(rows, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    col_tile = ct.reshape(cols, (1, BLOCK_C)) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)

    tvals = ct.astype(ct.gather(temp_ptr, (row_tile, col_tile), mask=mask2, padding_value=0),
                       ct.float32)
    rounded_bf = ct.astype(tvals, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf, ct.float32)
    ct.scatter(out0_ptr, (row_tile, col_tile), rounded_bf, mask=mask2)

    zeros2 = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    col_sums = ct.sum(ct.where(mask2, rounded_f, zeros2), axis=0)  # (BLOCK_C,)
    tile_row = ct.full((BLOCK_C,), row_group, dtype=ct.int32)
    ct.scatter(partials_ptr, (tile_row, cols), col_sums, mask=col_valid)


@ct.kernel
def _reduce_bf16_columns_kernel(
    source_ptr,      # bf16 [ROWS, HIDDEN]
    partials_ptr,    # f32  [NUM_GROUPS, HIDDEN]
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    col_group = ct.bid(1)
    rows = ct.arange(BLOCK_R, dtype=ct.int32) + row_group * BLOCK_R
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + col_group * BLOCK_C
    row_valid = rows < ROWS_
    col_valid = cols < HIDDEN_
    mask2 = ct.reshape(row_valid, (BLOCK_R, 1)) & ct.reshape(col_valid, (1, BLOCK_C))
    row_tile = ct.reshape(rows, (BLOCK_R, 1)) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    col_tile = ct.reshape(cols, (1, BLOCK_C)) + ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    vals = ct.astype(
        ct.gather(source_ptr, (row_tile, col_tile), mask=mask2, padding_value=0),
        ct.float32,
    )
    zeros2 = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    col_sum = ct.sum(ct.where(mask2, vals, zeros2), axis=0)
    tile_row = ct.full((BLOCK_C,), row_group, dtype=ct.int32)
    ct.scatter(partials_ptr, (tile_row, cols), col_sum, mask=col_valid)


@ct.kernel
def _finalize_bf16_sum_kernel(
    partials_ptr,    # f32 [NUM_GROUPS, HIDDEN]
    out_ptr,         # f32 [HIDDEN]
    NUM_GROUPS: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + col_block * BLOCK_C
    groups = ct.arange(GROUP_BLOCK, dtype=ct.int32)
    col_valid = cols < HIDDEN_
    grp_valid = groups < NUM_GROUPS
    mask2 = ct.reshape(grp_valid, (GROUP_BLOCK, 1)) & ct.reshape(col_valid, (1, BLOCK_C))
    grp_tile = ct.reshape(groups, (GROUP_BLOCK, 1)) + ct.zeros((GROUP_BLOCK, BLOCK_C), dtype=ct.int32)
    col_tile = ct.reshape(cols, (1, BLOCK_C)) + ct.zeros((GROUP_BLOCK, BLOCK_C), dtype=ct.int32)
    vals = ct.astype(
        ct.gather(partials_ptr, (grp_tile, col_tile), mask=mask2, padding_value=0),
        ct.float32,
    )
    zeros2 = ct.zeros((GROUP_BLOCK, BLOCK_C), dtype=ct.float32)
    tot = ct.sum(ct.where(mask2, vals, zeros2), axis=0)
    rounded = ct.astype(ct.astype(tot, ct.bfloat16), ct.float32)
    ct.scatter(out_ptr, cols, rounded, mask=col_valid)


@oracle_impl(hardware="B200", point="00333af8")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     *_shape_params) = inputs

    device = arg1_1.device
    full = torch.empty((BATCH, OUT0_HIDDEN, PADDED_TOKENS, 1), device=device, dtype=torch.float32)
    temp = torch.empty((OUT0_ROWS, OUT0_HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty((OUT0_ROWS, OUT0_HIDDEN), device=device, dtype=torch.bfloat16)
    out3 = torch.empty((OUT0_ROWS, OUT3_HIDDEN), device=device, dtype=torch.bfloat16)
    out2 = torch.empty((OUT0_HIDDEN,), device=device, dtype=torch.float32)
    out5 = torch.empty((OUT3_HIDDEN,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    zero_block = 256
    zero_grid = (max(FULL_NUMEL, TEMP_NUMEL) + zero_block - 1) // zero_block
    ct.launch(
        stream,
        (zero_grid, 1, 1),
        _zero_full_and_temp_kernel,
        (full.reshape(-1), temp.reshape(-1), zero_block),
    )

    # Views for cuTile access
    shift_1d = arg2_1.reshape(-1)
    denom_1d = arg3_1.reshape(-1)
    arg4_2d = arg4_1.reshape(PROB_ROWS, ARG4_CHANNELS)
    arg5_2d = arg5_1.reshape(PROB_ROWS, PROB_K)
    index_2d = arg6_1.reshape(PROB_K, TOKENS)

    BLOCK_C = 32
    BLOCK_K = 16
    ct.launch(
        stream,
        (PROB_ROWS, (ARG4_CHANNELS + BLOCK_C - 1) // BLOCK_C, 1),
        _scatter_probabilities_kernel,
        (arg0_1, arg1_1, shift_1d, denom_1d, arg4_2d, index_2d, temp,
         BLOCK_C, BLOCK_K),
    )

    reduce_block_r = 16
    reduce_block_c = 64
    out0_groups = (OUT0_ROWS + reduce_block_r - 1) // reduce_block_r
    out0_partials = torch.empty((out0_groups, OUT0_HIDDEN), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (out0_groups, (OUT0_HIDDEN + reduce_block_c - 1) // reduce_block_c, 1),
        _finalize_out0_partials_kernel,
        (temp, out0, out0_partials, reduce_block_r, reduce_block_c),
    )
    # Final BF16 sum for out2. NUM_GROUPS = out0_groups.
    group_block = 1
    while group_block < out0_groups:
        group_block *= 2
    ct.launch(
        stream,
        ((OUT0_HIDDEN + 16 - 1) // 16, 1, 1),
        _finalize_bf16_sum_kernel,
        (out0_partials, out2, out0_groups, OUT0_HIDDEN, group_block, 16),
    )

    # out3 computation
    BLOCK_R3 = 16
    ct.launch(
        stream,
        ((PROB_ROWS + BLOCK_R3 - 1) // BLOCK_R3, 1, 1),
        _store_out3_kernel,
        (arg0_1, arg1_1, shift_1d, denom_1d, arg5_2d, out3, BLOCK_R3, BLOCK_K),
    )

    # Column reduce out3
    out3_groups = (OUT0_ROWS + reduce_block_r - 1) // reduce_block_r
    out3_partials = torch.empty((out3_groups, OUT3_HIDDEN), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (out3_groups, (OUT3_HIDDEN + reduce_block_c - 1) // reduce_block_c, 1),
        _reduce_bf16_columns_kernel,
        (out3, out3_partials, OUT0_ROWS, OUT3_HIDDEN, reduce_block_r, reduce_block_c),
    )
    group_block3 = 1
    while group_block3 < out3_groups:
        group_block3 *= 2
    ct.launch(
        stream,
        ((OUT3_HIDDEN + 16 - 1) // 16, 1, 1),
        _finalize_bf16_sum_kernel,
        (out3_partials, out5, out3_groups, OUT3_HIDDEN, group_block3, 16),
    )

    return full, out0, out0.permute(1, 0), out2, out3, out3.permute(1, 0), out5
