"""cuTile port of sum_fc6414493f21: DeBERTa attention head layout + col sum.

Input x [BATCH*HEADS=192, QUERY=64, KEY=512] bf16.
Output out [ROWS=4096=BATCH*KEY, FEATURES=1536=HEADS*QUERY] bf16, where
  out[b*KEY+k, h*QUERY+q] = (x[b*HEADS+h, q, k] * 0.125).to(bf16)
Also out_sum [FEATURES] = bf16-rounded fp32 col sum of `out`.
Also full = 8.0 bf16 scalar.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
HEADS = 24
QUERY = 64
KEY = 512
ROWS = BATCH * KEY
FEATURES = HEADS * QUERY


@ct.kernel
def _layout_kernel(
    x_ptr,       # bf16 [BATCH, HEADS, QUERY, KEY]
    out_ptr,     # bf16 [BATCH, KEY, HEADS, QUERY]
    BLOCK_K: ct.Constant[int],
    BLOCK_Q: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    k_blk = ct.bid(2)  # walks KEY in units of BLOCK_K

    # Load x[b, h, :, k_blk*BLOCK_K:(k_blk+1)*BLOCK_K] shape (1, 1, QUERY, BLOCK_K)
    values = ct.load(x_ptr, index=(b, h, 0, k_blk), shape=(1, 1, QUERY, BLOCK_K))
    values = ct.reshape(values, (QUERY, BLOCK_K))
    scaled = ct.astype(ct.astype(values, ct.float32) * 0.125, ct.bfloat16)
    # transpose -> (BLOCK_K, QUERY) so that store to out[b, k_blk*BLOCK_K:, h, :]
    # writes rows indexed by (b, k) at column [h*QUERY:h*QUERY+QUERY].
    scaled_t = ct.transpose(scaled)
    scaled_t = ct.reshape(scaled_t, (1, BLOCK_K, 1, QUERY))
    ct.store(out_ptr, index=(b, k_blk, h, 0), tile=scaled_t)


@ct.kernel
def _colsum_kernel(
    out_ptr,     # bf16 [ROWS, FEATURES]
    out_sum_ptr, # f32 [FEATURES]
    ROWS_C: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
):
    f_blk = ct.bid(0)
    accum = ct.zeros(shape=(BLOCK_ROWS,), dtype=ct.float32)
    # Iterate over row blocks (compile-time chunks) via unrolled loop.
    # We do the reduction in 2D tiles.
    # Simple version: one shot load of ROWS x BLOCK_F for a small BLOCK_F.
    # Actually 4096 x 32 = 131072 tile elements; may exceed limits. Use loop.


@ct.kernel
def _colsum_kernel_row_partials(
    out_ptr,       # bf16 [ROWS, FEATURES]
    partial_ptr,   # f32 [NUM_ROW_TILES, FEATURES]
    ROWS_C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_F: ct.Constant[int],
):
    row_blk = ct.bid(0)
    f_blk = ct.bid(1)
    tile = ct.load(out_ptr, index=(row_blk, f_blk), shape=(BLOCK_M, BLOCK_F))
    partial = ct.sum(ct.astype(tile, ct.float32), axis=0)
    ct.store(partial_ptr, index=(row_blk, f_blk), tile=ct.reshape(partial, (1, BLOCK_F)))


@ct.kernel
def _colsum_finalize_kernel(
    partial_ptr,   # f32 [NUM_ROW_TILES, FEATURES]
    out_sum_ptr,   # f32 [FEATURES]
    NUM_ROW_TILES: ct.Constant[int],
    BLOCK_F: ct.Constant[int],
):
    f_blk = ct.bid(0)
    tile = ct.load(partial_ptr, index=(0, f_blk), shape=(NUM_ROW_TILES, BLOCK_F))
    total = ct.sum(tile, axis=0)
    # bf16 round-trip
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_sum_ptr, index=(f_blk,), tile=rounded)


@ct.kernel
def _store_scalar_full_kernel(full_ptr):
    ct.store(full_ptr, index=(0,), tile=ct.full(shape=(1,), fill_value=8.0, dtype=ct.bfloat16))


@oracle_impl(
    hardware="B200",
    point="197ee996",
    BLOCK_M=64,
    BLOCK_N=64,
    FINAL_BLOCK_N=64,
)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, FINAL_BLOCK_N):
    (x, *_shape_params) = inputs
    device = x.device

    full = torch.full((), 8.0, dtype=torch.bfloat16)  # CPU per repro
    out = torch.empty_strided((ROWS, FEATURES), (FEATURES, 1), device=device, dtype=torch.bfloat16)
    out_sum = torch.empty_strided((FEATURES,), (1,), device=device, dtype=torch.float32)

    # x is [192, 64, 512]; view as [BATCH, HEADS, QUERY, KEY]
    x_4d = x.view(BATCH, HEADS, QUERY, KEY)
    # out is [ROWS=BATCH*KEY, FEATURES=HEADS*QUERY]; view as [BATCH, KEY, HEADS, QUERY]
    out_4d = out.view(BATCH, KEY, HEADS, QUERY)

    stream = torch.cuda.current_stream()
    BLOCK_K = 32
    ct.launch(
        stream,
        (BATCH, HEADS, KEY // BLOCK_K),
        _layout_kernel,
        (x_4d, out_4d, BLOCK_K, QUERY),
    )

    # Column sum via row-partial + finalize
    BLOCK_M = 128
    BLOCK_F = 64
    NUM_ROW_TILES = ROWS // BLOCK_M
    partial = torch.empty(
        (NUM_ROW_TILES, FEATURES), device=device, dtype=torch.float32
    )
    ct.launch(
        stream,
        (NUM_ROW_TILES, FEATURES // BLOCK_F, 1),
        _colsum_kernel_row_partials,
        (out, partial, ROWS, BLOCK_M, BLOCK_F),
    )
    ct.launch(
        stream,
        (FEATURES // BLOCK_F, 1, 1),
        _colsum_finalize_kernel,
        (partial, out_sum, NUM_ROW_TILES, BLOCK_F),
    )

    return full, out, out.permute(1, 0), out_sum
