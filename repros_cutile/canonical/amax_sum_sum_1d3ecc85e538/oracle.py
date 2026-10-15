"""cuTile port of amax_sum_sum_1d3ecc85e538 (NEW_PATTERN): MobileBERT biased XENT.

Two cuTile kernels mirror the Triton oracle:
1. Per-row: in-kernel bias+logits materialization (bf16 output side-store) +
   online amax + log-sum-exp over N_COLS, bf16 log-softmax roundtrip on target,
   per-row loss + valid_mask.
2. Scalar: reduce loss sum / valid_count with bf16 boundary into scalar loss.
Target gather (row * ROW_STRIDE + safe_label) hoisted to torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _biased_xent_rows_kernel(
    logits_ptr,        # bf16 [N_ROWS, LOGITS_STRIDE]
    bias_ptr,          # bf16 [LOGITS_STRIDE] (only first N_COLS valid)
    targets_ptr,       # f32 [N_ROWS] pre-computed (target_logit + target_bias) bf16-rounded
    is_valid_ptr,      # f32 [N_ROWS] 1.0/0.0
    logits_out_ptr,    # bf16 [N_ROWS, N_COLS]  biased logits output (contiguous)
    loss_ptr,          # bf16 [N_ROWS]
    valid_out_ptr,     # f32 [N_ROWS]
    N_COLS: ct.Constant[int],
    N_TILES: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    target_scalar = ct.reshape(
        ct.astype(ct.load(targets_ptr, index=(row,), shape=(1,)), ct.float32),
        (),
    )
    is_valid_scalar = ct.reshape(
        ct.load(is_valid_ptr, index=(row,), shape=(1,)),
        (),
    ) > 0.0

    row_max = ct.astype(-float("inf"), ct.float32)
    denom = ct.astype(0.0, ct.float32)
    for tile_i in ct.static_iter(range(N_TILES)):
        block_start = tile_i * BLOCK_N
        # Load raw logits (stride LOGITS_STRIDE).
        logits = ct.load(
            logits_ptr, index=(row, tile_i), shape=(1, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        # Load bias (contiguous stride).
        bias = ct.load(
            bias_ptr, index=(tile_i,), shape=(BLOCK_N,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        logits_f = ct.astype(logits, ct.float32)
        bias_f = ct.astype(bias, ct.float32)
        bias_2d = ct.reshape(bias_f, (1, BLOCK_N))
        biased_bf = ct.astype(logits_f + bias_2d, ct.bfloat16)

        # Store biased bf16 output — mask needed for tail cols.
        cols = ct.arange(BLOCK_N, dtype=ct.int32) + block_start
        col_valid = cols < N_COLS
        col_valid_2d = ct.reshape(col_valid, (1, BLOCK_N))
        # Use scatter with the linearized row*N_COLS + cols index
        row_idx = ct.full((BLOCK_N,), row, dtype=ct.int32)
        offs = row_idx * N_COLS + cols
        biased_1d = ct.reshape(biased_bf, (BLOCK_N,))
        ct.scatter(logits_out_ptr, (offs,), biased_1d, mask=col_valid)

        # Online amax + log-sum-exp
        biased_f = ct.astype(biased_bf, ct.float32)
        neg_inf = ct.full((1, BLOCK_N), -float("inf"), dtype=ct.float32)
        biased_masked = ct.where(col_valid_2d, biased_f, neg_inf)
        block_max = ct.max(biased_masked)
        new_max = ct.maximum(row_max, block_max)
        denom = denom * ct.exp(row_max - new_max)
        denom = denom + ct.sum(ct.exp(biased_masked - new_max))
        row_max = new_max

    log_denom = ct.log(denom)
    logp = target_scalar - row_max - log_denom
    # bf16 log-softmax rounding on the loss (matches Triton's `.to(bf16).to(f32)`)
    loss_val = 0.0 - ct.astype(ct.astype(logp, ct.bfloat16), ct.float32)
    zero_f = ct.astype(0.0, ct.float32)
    one_f = ct.astype(1.0, ct.float32)
    loss_masked = ct.where(is_valid_scalar, loss_val, zero_f)
    valid_val = ct.where(is_valid_scalar, one_f, zero_f)
    # loss stored as bf16 (matches Triton)
    ct.store(
        loss_ptr, index=(row,),
        tile=ct.astype(ct.reshape(loss_masked, (1,)), ct.bfloat16),
    )
    ct.store(valid_out_ptr, index=(row,), tile=ct.reshape(valid_val, (1,)))


@ct.kernel
def _mean_reduce_kernel(
    loss_ptr,        # bf16 [N_ROWS]
    valid_ptr,       # f32  [N_ROWS]
    out_ptr,         # bf16 [1]
    N_ROWS_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    cols = ct.arange(BLOCK_M, dtype=ct.int32)
    mask = cols < N_ROWS_
    losses = ct.astype(
        ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32,
    )
    valid = ct.load(valid_ptr, index=(0,), shape=(BLOCK_M,),
                    padding_mode=ct.PaddingMode.ZERO)
    zero_f = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    losses_m = ct.where(mask, losses, zero_f)
    valid_m = ct.where(mask, valid, zero_f)
    total_loss = ct.astype(ct.astype(ct.sum(losses_m), ct.bfloat16), ct.float32)
    total_valid = ct.astype(ct.astype(ct.sum(valid_m), ct.bfloat16), ct.float32)
    ct.store(
        out_ptr, index=(0,),
        tile=ct.astype(ct.reshape(total_loss / total_valid, (1,)), ct.bfloat16),
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(n):
    r = 1
    while r < n:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="6f08aca4")
def oracle_forward(inputs):
    labels, logits, bias, shape_3d, shape_2d, output_shape = inputs
    del shape_2d
    n_rows = int(labels.numel())
    n_cols = int(bias.numel())
    device = logits.device
    out_shape = _shape_tuple(output_shape)

    labels_flat = labels.view(-1)
    is_valid = labels_flat != -100
    is_valid_f = is_valid.to(torch.float32)
    safe_labels = torch.where(is_valid, labels_flat, torch.zeros_like(labels_flat))
    # target_logit = logits[row, safe_label], target_bias = bias[safe_label],
    # target = round_bf16(logit + bias)
    logits_slice_f = logits[:, :n_cols].to(torch.float32)
    bias_f = bias.to(torch.float32)
    target_logit = torch.gather(
        logits_slice_f, 1,
        safe_labels.unsqueeze(1).clamp(0, n_cols - 1),
    ).squeeze(1)
    target_bias = bias_f[safe_labels.clamp(0, n_cols - 1)]
    targets = (target_logit + target_bias).to(torch.bfloat16).to(torch.float32)

    logits_out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=device, dtype=torch.bfloat16,
    )
    loss_per_row = torch.empty((n_rows,), device=device, dtype=torch.bfloat16)
    valid_per_row = torch.empty((n_rows,), device=device, dtype=torch.float32)
    out = torch.empty_strided((), (), device=device, dtype=torch.bfloat16)

    # BLOCK_N should divide logits row stride evenly (or use padding).
    # LOGITS_STRIDE=30528, BLOCK_N=1024, n_tiles=30 covers cols 0..30720.
    BLOCK_N = 1024
    n_tiles = (n_cols + BLOCK_N - 1) // BLOCK_N

    # For bias, we view it as flat length >= n_tiles * BLOCK_N with zero pad
    # via the padding_mode=ZERO in the kernel.

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _biased_xent_rows_kernel,
        (logits, bias, targets, is_valid_f,
         logits_out.view(n_rows * n_cols),
         loss_per_row, valid_per_row,
         n_cols, n_tiles, BLOCK_N),
    )
    block_m = _next_pow2(n_rows)
    ct.launch(
        stream, (1, 1, 1), _mean_reduce_kernel,
        (loss_per_row, valid_per_row, out.view(1), n_rows, block_m),
    )
    return out, logits_out.view(_shape_tuple(shape_3d))
