"""cuTile port of amax_sum_ac21f557e1ce: XGLM causal-mask bf16 attention softmax.

Two outputs: bf16 causal mask [32,1,128,128] and bf16 softmax probabilities.
Uses BLOCK_M > 1 to batch queries per program, matching the Triton reference.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BF16_MIN = -3.3895313892515355e38


@ct.kernel
def _causal_mask_kernel(
    mask_out,   # (B, 1, Q, K) bf16
    Q: ct.Constant[int],
    K: ct.Constant[int],
    FILL: ct.Constant[float],
    BLOCK_M: ct.Constant[int],
    Q_TILES: ct.Constant[int],
):
    b = ct.bid(0)
    q_block = ct.bid(1)
    # BLOCK_M consecutive queries.
    rows = ct.arange(BLOCK_M, dtype=ct.int32) + q_block * BLOCK_M
    cols = ct.arange(K, dtype=ct.int32)
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, K))
    rows_broad = ct.broadcast_to(rows_2d, (BLOCK_M, K))
    cols_broad = ct.broadcast_to(cols_2d, (BLOCK_M, K))
    causal = cols_broad <= rows_broad
    mask_2d = ct.where(
        causal,
        ct.zeros(shape=(BLOCK_M, K), dtype=ct.float32),
        ct.full(shape=(BLOCK_M, K), fill_value=FILL, dtype=ct.float32),
    )
    mask_bf16 = ct.astype(mask_2d, ct.bfloat16)
    tile = ct.reshape(mask_bf16, (1, 1, BLOCK_M, K))
    ct.store(mask_out, index=(b, 0, q_block, 0), tile=tile)


@ct.kernel
def _softmax_kernel(
    x,          # (N, K) bf16 - flat softmax input
    out,        # (N, K) bf16
    N: ct.Constant[int],
    K: ct.Constant[int],
    Q: ct.Constant[int],
    FILL: ct.Constant[float],
    BLOCK_M: ct.Constant[int],
    Q_TILES: ct.Constant[int],
):
    row_block = ct.bid(0)  # tile index over N rows
    # Determine the query index (mod Q) for each row in this block.
    # rows = row_block * BLOCK_M + arange(BLOCK_M); q = rows % Q.
    # Since BLOCK_M divides Q, all rows in one block belong to the same
    # (batch, head), and the q values are consecutive within [0, Q).
    q_tile = row_block - (row_block // Q_TILES) * Q_TILES
    q_start = q_tile * BLOCK_M
    q_offsets = ct.arange(BLOCK_M, dtype=ct.int32) + q_start
    cols = ct.arange(K, dtype=ct.int32)
    q_2d = ct.reshape(q_offsets, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, K))
    q_broad = ct.broadcast_to(q_2d, (BLOCK_M, K))
    cols_broad = ct.broadcast_to(cols_2d, (BLOCK_M, K))
    causal = cols_broad <= q_broad
    mask_f = ct.where(
        causal,
        ct.zeros(shape=(BLOCK_M, K), dtype=ct.float32),
        ct.full(shape=(BLOCK_M, K), fill_value=FILL, dtype=ct.float32),
    )
    mask_bf16 = ct.astype(mask_f, ct.bfloat16)
    mask_f = ct.astype(mask_bf16, ct.float32)

    x_tile = ct.load(x, index=(row_block, 0), shape=(BLOCK_M, K))
    x_f = ct.astype(x_tile, ct.float32)
    added_bf16 = ct.astype(x_f + mask_f, ct.bfloat16)
    added_f = ct.astype(added_bf16, ct.float32)
    fill_tile = ct.full(shape=(BLOCK_M, K), fill_value=FILL, dtype=ct.float32)
    clamped_bf16 = ct.astype(ct.where(added_f > fill_tile, added_f, fill_tile), ct.bfloat16)
    scores = ct.astype(clamped_bf16, ct.float32)

    row_max = ct.max(scores, axis=1, keepdims=True)
    has_nan_tile = scores != scores
    has_nan = ct.max(ct.astype(has_nan_tile, ct.float32), axis=1, keepdims=True) > 0.0
    row_max_final = ct.where(has_nan, ct.full(shape=(BLOCK_M, 1), fill_value=float('nan'), dtype=ct.float32), row_max)
    numer = ct.exp(scores - row_max_final)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(out, index=(row_block, 0), tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_M=16)
def oracle_forward(inputs, *, BLOCK_M: int):
    arg0_1, shape0, shape1, shape2 = inputs
    # shape0: [-1, 16, 128, 128] resolved => [B, H, Q, K]
    # arg0_1 is originally [512, 128, 128] = [B*H, Q, K]
    dims = [int(d) for d in shape0]
    numel = arg0_1.numel()
    known = 1
    missing = -1
    for i, d in enumerate(dims):
        if d == -1:
            missing = i
        else:
            known *= d
    if missing >= 0:
        dims[missing] = numel // known
    B, H, Q, K = dims

    # mask_shape from shape1 is [B, 1, Q, K]; -1 dims default to 1
    mask_shape = [1 if int(d) == -1 else int(d) for d in shape1]
    mask = torch.empty_strided(
        tuple(mask_shape),
        (mask_shape[1] * mask_shape[2] * mask_shape[3],
         mask_shape[2] * mask_shape[3],
         mask_shape[3], 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    out_shape = [int(d) for d in shape2]
    if -1 in out_shape:
        idx = out_shape.index(-1)
        prod = 1
        for i, dd in enumerate(out_shape):
            if i != idx:
                prod *= dd
        out_shape[idx] = numel // prod
    out = torch.empty_strided(
        tuple(out_shape),
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )

    assert Q % BLOCK_M == 0, f"BLOCK_M={BLOCK_M} must divide Q={Q}"
    q_tiles = Q // BLOCK_M

    stream = torch.cuda.current_stream()
    # Kernel 1: build mask, BLOCK_M query rows per program.
    ct.launch(stream, (B, q_tiles, 1), _causal_mask_kernel,
              (mask, Q, K, BF16_MIN, BLOCK_M, q_tiles))
    # Kernel 2: softmax with per-row causal mask baked in.
    x_2d = arg0_1.view(-1, K)  # (N, K)
    N = x_2d.shape[0]
    out_2d = out.view(-1, K)
    ct.launch(stream, (ct.cdiv(N, BLOCK_M), 1, 1), _softmax_kernel,
              (x_2d, out_2d, N, K, Q, BF16_MIN, BLOCK_M, q_tiles))
    return mask, out
