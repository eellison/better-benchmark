"""cuTile port of amax_sum_sum_4779980113db (NEW_PATTERN): bf16 ignore-index
cross-entropy mean with logits view alias.

Ports the Triton kernel — the inline PTX sub.rn/mul.rn/etc. simply enforce
IEEE 754 RNE which is cuTile's default. `.to(bf16).to(f32)` roundtrips
survive as `ct.astype`.

For each row: online logsumexp over vocab (block-tiled), gather target logit,
compute `-(target - row_max - log(denom))` rounded to bf16. Then a final
kernel sums per-row losses & valid counts, divides them (bf16 rounded)
producing the scalar mean.

Target-index gather: precompute `row*N_COLS + safe_label` outside the kernel
and pass it in.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xent_row_kernel(
    labels_ptr,       # i64 [n_rows]
    logits_flat_ptr,  # bf16 [n_rows * n_cols]
    target_idx_ptr,   # i32 [n_rows]  = row*N_COLS + safe_label
    is_valid_ptr,     # uint8 [n_rows]  = label != -100
    loss_ptr,         # bf16 [n_rows]
    valid_ptr,        # f32 [n_rows]
    N_COLS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    # Online logsumexp: loop over columns in BLOCK_N chunks.
    row_max = ct.astype(-3.4e38, ct.float32)
    denom = ct.astype(0.0, ct.float32)
    n_blocks = ct.cdiv(N_COLS, BLOCK_N)
    row_i32 = ct.astype(row, ct.int32)

    for i in range(n_blocks):
        # Build tile of column offsets from `row*N_COLS + i*BLOCK_N` to
        # `row*N_COLS + (i+1)*BLOCK_N`.
        offs = row_i32 * N_COLS + i * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
        col_idx = i * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
        col_mask = col_idx < N_COLS
        logits = ct.gather(logits_flat_ptr, offs)
        logits_f = ct.astype(logits, ct.float32)
        # Mask OOB to -inf.
        neg_inf_tile = ct.full((BLOCK_N,), -3.4e38, dtype=ct.float32)
        masked = ct.where(col_mask, logits_f, neg_inf_tile)
        block_max = ct.max(masked)
        new_max = ct.maximum(row_max, block_max)
        # Rescale denom.
        denom = denom * ct.exp(row_max - new_max)
        exp_vals = ct.exp(masked - new_max)
        zero_tile = ct.full((BLOCK_N,), 0.0, dtype=ct.float32)
        exp_masked = ct.where(col_mask, exp_vals, zero_tile)
        denom = denom + ct.sum(exp_masked)
        row_max = new_max

    log_denom = ct.log(denom)

    # Load target logit as a 1-tile.
    target_idx_tile = ct.load(target_idx_ptr, index=(row,), shape=(1,))
    target_i32 = ct.astype(target_idx_tile, ct.int32)
    target_tile = ct.gather(logits_flat_ptr, target_i32)
    target_f = ct.astype(target_tile, ct.float32)  # shape (1,)

    # Compute in scalar (broadcasting) — row_max and log_denom are scalars.
    row_max_1 = ct.full((1,), row_max, dtype=ct.float32)
    log_denom_1 = ct.full((1,), log_denom, dtype=ct.float32)
    logp = target_f - row_max_1 - log_denom_1  # shape (1,)
    logp_bf16 = ct.astype(logp, ct.bfloat16)
    logp_rounded = ct.astype(logp_bf16, ct.float32)
    loss_val = ct.astype(0.0, ct.float32) - logp_rounded

    # Load is_valid as a 1-tile.
    valid_tile = ct.load(is_valid_ptr, index=(row,), shape=(1,))
    zero_u8 = ct.full((1,), 0, dtype=ct.uint8)
    valid_bool = valid_tile != zero_u8
    zero_f = ct.full((1,), 0.0, dtype=ct.float32)
    one_f = ct.full((1,), 1.0, dtype=ct.float32)
    loss_masked = ct.where(valid_bool, loss_val, zero_f)
    valid_masked = ct.where(valid_bool, one_f, zero_f)

    loss_bf16 = ct.astype(loss_masked, ct.bfloat16)
    ct.store(loss_ptr, index=(row,), tile=loss_bf16)
    ct.store(valid_ptr, index=(row,), tile=valid_masked)


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,       # bf16 [n_rows]
    valid_ptr,      # f32 [n_rows]
    out_ptr,        # bf16 [1]
    N_ROWS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    losses = ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    valids = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                     padding_mode=ct.PaddingMode.ZERO)
    losses_f = ct.astype(losses, ct.float32)
    # Also need to mask valids beyond N_ROWS (they're loaded as 0 which is fine
    # for count).
    idx = ct.arange(BLOCK_M, dtype=ct.int32)
    mask = idx < N_ROWS
    losses_f = ct.where(mask, losses_f, 0.0)
    valids = ct.where(mask, valids, 0.0)

    total_loss = ct.sum(losses_f)
    total_loss_bf16 = ct.astype(total_loss, ct.bfloat16)
    total_loss_f = ct.astype(total_loss_bf16, ct.float32)

    total_valid = ct.sum(valids)
    total_valid_bf16 = ct.astype(total_valid, ct.bfloat16)
    total_valid_f = ct.astype(total_valid_bf16, ct.float32)

    result = total_loss_f / total_valid_f
    result_bf16 = ct.astype(ct.full((1,), result, dtype=ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(0,), tile=result_bf16)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="6098747b", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="bba6ebb5", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="4f6a8ed5", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="29bccbec", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="f7a45f0f", BLOCK_N=8192)
@oracle_impl(hardware="B200", point="8e79bb3c", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N):
    labels, logits, shape_3d, shape_2d = inputs
    view_shape = _shape_tuple(shape_3d)
    n_rows = int(labels.numel())
    n_cols = int(shape_2d[1])

    logits_view = logits.view(view_shape)
    logits_2d = logits.view(n_rows, n_cols)
    logits_flat = logits.reshape(-1)
    labels_1d = labels.view(-1)

    # Precompute target_idx = row*N_COLS + safe_label, and is_valid.
    row_arr = torch.arange(n_rows, device=logits.device, dtype=torch.int64)
    is_valid_i = (labels_1d != -100)
    # Use clamp instead of allocating a torch.zeros_like buffer for the branch.
    safe_label = labels_1d.clamp(min=0)
    target_idx = (row_arr * n_cols + safe_label).to(torch.int32)
    is_valid_u8 = is_valid_i.to(torch.uint8)

    loss_per_row = torch.empty_strided(
        (n_rows,), (1,), device=logits.device, dtype=torch.bfloat16,
    )
    valid_per_row = torch.empty_strided(
        (n_rows,), (1,), device=logits.device, dtype=torch.float32,
    )
    div = torch.empty_strided((1,), (1,), device=logits.device, dtype=torch.bfloat16)

    block_n = min(BLOCK_N, _next_power_of_2(n_cols))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _xent_row_kernel,
        (labels_1d, logits_flat, target_idx, is_valid_u8, loss_per_row,
         valid_per_row, n_cols, block_n),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _mean_reduce_kernel,
        (loss_per_row, valid_per_row, div, n_rows, _next_power_of_2(n_rows)),
    )
    return logits_view, div.view(())
