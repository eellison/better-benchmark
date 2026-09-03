"""cuTile port of amax_sum_sum_671a76fab586 (NEW_PATTERN): static large
vocabulary cross-entropy.

Two cuTile kernels mirror the Triton oracle:
1. Per-row: online amax + log-sum-exp over K=262144 in-kernel (streaming over
   32 tiles of BLOCK_N=8192), bf16 log-softmax roundtrip on target, per-row
   loss (bf16).
2. Scalar: reduce loss sum over N_ROWS=8192 rows (bf16 output).
Target gather (row * COLS + safe_label) hoisted to torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_ROWS = 8192
N_COLS = 262144


@ct.kernel
def _xent_rows_kernel(
    logits_ptr,      # bf16 [N_ROWS, N_COLS]
    targets_ptr,     # f32 [N_ROWS] pre-gathered target logits
    is_valid_ptr,    # f32 [N_ROWS] 1.0/0.0
    amax_out_ptr,    # f32 [N_ROWS]
    log_out_ptr,     # f32 [N_ROWS]
    loss_ptr,        # bf16 [N_ROWS]
    BLOCK_N_: ct.Constant[int],
    NUM_BLOCKS_: ct.Constant[int],
):
    row = ct.bid(0)
    target = ct.astype(
        ct.load(targets_ptr, index=(row,), shape=(1,)),
        ct.float32,
    )
    target_scalar = ct.reshape(target, ())
    is_valid_v = ct.load(is_valid_ptr, index=(row,), shape=(1,))
    is_valid_scalar = ct.reshape(is_valid_v, ()) > 0.0

    row_max = ct.astype(-float("inf"), ct.float32)
    denom = ct.astype(0.0, ct.float32)
    for b in ct.static_iter(range(NUM_BLOCKS_)):
        x_bf = ct.load(logits_ptr, index=(row, b), shape=(1, BLOCK_N_))
        x_f = ct.astype(x_bf, ct.float32)
        block_max = ct.max(x_f)
        new_max = ct.maximum(row_max, block_max)
        denom = denom * ct.exp(row_max - new_max)
        denom = denom + ct.sum(ct.exp(x_f - new_max))
        row_max = new_max

    log_denom = ct.log(denom)
    logp = target_scalar - row_max - log_denom
    # bf16 log-softmax rounding on loss (matches Triton `.to(bf16).to(f32)`)
    loss_val = 0.0 - ct.astype(ct.astype(logp, ct.bfloat16), ct.float32)
    zero_f = ct.astype(0.0, ct.float32)
    loss_masked = ct.where(is_valid_scalar, loss_val, zero_f)

    ct.store(amax_out_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(log_out_ptr, index=(row,), tile=ct.reshape(log_denom, (1,)))
    ct.store(
        loss_ptr, index=(row,),
        tile=ct.astype(ct.reshape(loss_masked, (1,)), ct.bfloat16),
    )


@ct.kernel
def _loss_reduce_kernel(
    loss_ptr,    # bf16 [N_ROWS]
    out_ptr,     # bf16 [1]
    N_ROWS_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    losses = ct.astype(
        ct.load(loss_ptr, index=(0,), shape=(BLOCK_M,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32,
    )
    total = ct.sum(losses)
    ct.store(out_ptr, index=(0,),
             tile=ct.astype(ct.reshape(total, (1,)), ct.bfloat16))


def _next_pow2(n):
    r = 1
    while r < n:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="b899b223", BLOCK_N=8192)
def oracle_forward(inputs, *, BLOCK_N):
    logits, labels = inputs
    rows = int(logits.shape[0])
    cols = int(logits.shape[1])
    device = logits.device

    is_valid = labels != -100
    is_valid_f = is_valid.to(torch.float32)
    safe_labels = torch.where(is_valid, labels, torch.zeros_like(labels))
    targets = torch.gather(
        logits.to(torch.float32), 1,
        safe_labels.unsqueeze(1).clamp(0, cols - 1),
    ).squeeze(1)

    amax_out = torch.empty_strided((rows, 1), (1, 1), device=device, dtype=torch.float32)
    log_out = torch.empty_strided((rows, 1), (1, 1), device=device, dtype=torch.float32)
    loss_per_row = torch.empty((rows,), device=device, dtype=torch.bfloat16)
    loss_out = torch.empty_strided((), (), device=device, dtype=torch.bfloat16)

    num_blocks = cols // BLOCK_N
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _xent_rows_kernel,
        (logits, targets, is_valid_f,
         amax_out.view(rows), log_out.view(rows), loss_per_row,
         BLOCK_N, num_blocks),
    )

    block_m = _next_pow2(rows)
    ct.launch(
        stream, (1, 1, 1), _loss_reduce_kernel,
        (loss_per_row, loss_out.view(1), rows, block_m),
    )
    return amax_out, log_out, loss_out
