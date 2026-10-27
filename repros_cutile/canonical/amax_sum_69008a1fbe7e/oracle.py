"""cuTile port of amax_sum_69008a1fbe7e: XGLM additive-bias attention softmax.

For each row of the flat bf16 `[512, 128]` scores tensor, add the broadcast
`[32, 1, 128, 128]` bias (viewed as [32, 128, 128]), clamp to -3.3895e38,
then stable fp32 softmax and cast back to bf16.

BLOCK_M=16, KEY=128: rows are laid out (batch, head, query). Each tile of
16 rows lies entirely within one (batch, head), and query iterates
consecutively — so the tile-space index into bias is
(batch, q_tile, 0) with tile shape (1, BLOCK_M, KEY).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
QUERY = 128
KEY = 128
ROWS = BATCH * HEADS * QUERY
SENTINEL = -3.3895313892515355e38


@ct.kernel
def _bias_softmax_kernel(
    scores_ptr,   # bf16 [ROWS, KEY]
    bias_ptr,     # bf16 [BATCH, QUERY, KEY]
    out_ptr,      # bf16 [ROWS, KEY]
    BLOCK_M: ct.Constant[int],
    HEADS_C: ct.Constant[int],
    QUERY_C: ct.Constant[int],
    KEY_C: ct.Constant[int],
    Q_TILES: ct.Constant[int],  # QUERY_C // BLOCK_M
):
    row_tile = ct.bid(0)
    # For row_tile, base_row = row_tile * BLOCK_M.
    # batch = base_row // (HEADS*QUERY)
    # q_base = base_row % QUERY  (always multiple of BLOCK_M)
    # bias tile-space partition (1, BLOCK_M, KEY):
    #   tile index dim 1 = q_base / BLOCK_M = row_tile % Q_TILES
    # But we need batch = row_tile / (HEADS * Q_TILES).
    batch = row_tile // (HEADS_C * Q_TILES)
    q_tile = row_tile - (row_tile // Q_TILES) * Q_TILES

    x = ct.load(scores_ptr, index=(row_tile, 0), shape=(BLOCK_M, KEY_C))
    x_f = ct.astype(x, ct.float32)

    bias = ct.load(bias_ptr, index=(batch, q_tile, 0),
                   shape=(1, BLOCK_M, KEY_C))
    bias = ct.reshape(bias, (BLOCK_M, KEY_C))
    bias_f = ct.astype(bias, ct.float32)

    # add + bf16 rounding boundary + maximum(sentinel)
    scores = ct.astype(ct.astype(x_f + bias_f, ct.bfloat16), ct.float32)
    scores = ct.maximum(scores, ct.full(shape=(BLOCK_M, KEY_C),
                                        fill_value=SENTINEL,
                                        dtype=ct.float32))
    row_max = ct.max(scores, axis=1, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(out_ptr, index=(row_tile, 0),
             tile=ct.astype(probs, ct.bfloat16))


@oracle_impl(hardware="B200", point="87c3ffc0", BLOCK_M=16)
def oracle_forward(inputs, *, BLOCK_M):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty_strided(
        (BATCH * HEADS, QUERY, KEY),
        (QUERY * KEY, KEY, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    scores_flat = arg0_1.view(ROWS, KEY)
    out_flat = out.view(ROWS, KEY)
    bias_view = arg1_1.view(BATCH, QUERY, KEY)
    stream = torch.cuda.current_stream()
    q_tiles = QUERY // BLOCK_M
    ct.launch(
        stream, (ROWS // BLOCK_M, 1, 1), _bias_softmax_kernel,
        (scores_flat, bias_view, out_flat,
         BLOCK_M, HEADS, QUERY, KEY, q_tiles),
    )
    return out
