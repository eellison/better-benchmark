"""cuTile port of sum_f24f665135f6: BERT select-scatter + dropout + column-sum.

Reference (BATCH=16, SEQ=128, FEATURES=768):
  input [2048, 768] bf16, bias [16, 768] bf16, mask0/mask1 [16, 128, 768] bool
  1. select_scatter of bias into rows where seq==0 (broadcast per batch)
  2. add input + scatter -> f32
  3. multiply by (mask0 * 1.1111111...) -> f32 out_f32
  4. bf16 round; multiply by (bf16(mask1) * 1.1111111...); round to bf16 out_bf16
  5. sum out_bf16 over rows (dim 0), bf16-round the partials before final f32 add
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
FEATURES = 768
ROWS = BATCH * SEQ
BLOCK_FEATURES = 32
ROW_BLOCK = 64
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _materialize_partial_kernel(
    input_ptr,        # bf16 [ROWS, FEATURES]
    bias_ptr,         # bf16 [BATCH, FEATURES]
    mask0_ptr,        # b8   [BATCH, SEQ, FEATURES]  contiguous
    mask1_ptr,        # b8   [BATCH, SEQ, FEATURES]  contiguous
    out_f32_ptr,      # f32 [BATCH, SEQ, FEATURES]
    out_bf16_ptr,     # bf16 [ROWS, FEATURES]
    partial_ptr,      # f32 [num_row_tiles, FEATURES]
    ROW_BLOCK_: ct.Constant[int],
    BLOCK_F: ct.Constant[int],
    SEQ_: ct.Constant[int],
):
    row_tile = ct.bid(0)
    feat_tile = ct.bid(1)

    # Load input tile of shape (ROW_BLOCK, BLOCK_F)
    x_bf = ct.load(input_ptr, index=(row_tile, feat_tile), shape=(ROW_BLOCK_, BLOCK_F))
    x = ct.astype(x_bf, ct.float32)

    # Determine which batch each row belongs to. ROW_BLOCK=64, SEQ=128 -> two
    # tiles cover a batch (2 * 64 = 128). row_tile_within_batch = row_tile mod 2.
    tile_start = row_tile * ROW_BLOCK_
    batch_id = tile_start // SEQ_
    seq_start_within_batch = tile_start - batch_id * SEQ_
    # Load bias[batch_id, feat_tile*BLOCK_F : ...]
    bias = ct.load(bias_ptr, index=(batch_id, feat_tile), shape=(1, BLOCK_F))
    bias_f = ct.astype(bias, ct.float32)
    bias_bc = ct.reshape(bias_f, (1, BLOCK_F))

    # Build a per-row mask: True iff seq_within_batch == 0.
    row_idx = ct.arange(ROW_BLOCK_, dtype=ct.int32)
    seq_within = ct.reshape(row_idx + seq_start_within_batch, (ROW_BLOCK_, 1))
    scatter_active = seq_within == 0

    scatter = ct.where(scatter_active, bias_bc, 0.0)
    first = x + scatter

    # mask0 is [B, S, F]; row_tile covers rows [row_tile*RB, (row_tile+1)*RB).
    # Load a slab of shape (ROW_BLOCK_, BLOCK_F) from flat [ROWS, F] view.
    mask0 = ct.load(mask0_ptr, index=(row_tile, feat_tile), shape=(ROW_BLOCK_, BLOCK_F))
    mask0_f = ct.astype(mask0, ct.float32)
    first_scale = mask0_f * DROPOUT_SCALE
    out_f32 = first * first_scale
    ct.store(out_f32_ptr, index=(row_tile, feat_tile), tile=out_f32)

    first_bf16 = ct.astype(out_f32, ct.bfloat16)
    mask1 = ct.load(mask1_ptr, index=(row_tile, feat_tile), shape=(ROW_BLOCK_, BLOCK_F))
    mask1_f = ct.astype(mask1, ct.float32)
    second_scale_bf = ct.astype(mask1_f * DROPOUT_SCALE, ct.bfloat16)
    second_scale_f = ct.astype(second_scale_bf, ct.float32)
    out_bf16_f = ct.astype(first_bf16, ct.float32) * second_scale_f
    out_bf16 = ct.astype(out_bf16_f, ct.bfloat16)
    ct.store(out_bf16_ptr, index=(row_tile, feat_tile), tile=out_bf16)

    partial = ct.sum(ct.astype(out_bf16, ct.float32), axis=0)
    ct.store(partial_ptr, index=(row_tile, feat_tile), tile=ct.reshape(partial, (1, BLOCK_F)))


@ct.kernel
def _finish_sum_kernel(
    partial_ptr,       # f32 [num_row_tiles, FEATURES]
    sum_ptr,           # f32 [FEATURES]
    ROW_TILES: ct.Constant[int],
    BLOCK_F: ct.Constant[int],
):
    feat_tile = ct.bid(0)
    tile = ct.load(partial_ptr, index=(0, feat_tile), shape=(ROW_TILES, BLOCK_F))
    total = ct.sum(tile, axis=0)
    total_bf16 = ct.astype(total, ct.bfloat16)
    total_f32 = ct.astype(total_bf16, ct.float32)
    ct.store(sum_ptr, index=(feat_tile,), tile=total_f32)


@oracle_impl(hardware="B200", point="2f9fa5d4")
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]
    device = arg0_1.device
    row_tiles = ROWS // ROW_BLOCK

    out_f32 = torch.empty_strided(
        (BATCH, SEQ, FEATURES),
        (SEQ * FEATURES, FEATURES, 1),
        device=device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (ROWS, FEATURES),
        (FEATURES, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty((row_tiles, FEATURES), device=device, dtype=torch.float32)
    sum_out = torch.empty((FEATURES,), device=device, dtype=torch.float32)

    # Flat views for cuTile row-tile indexing.
    mask0_flat = arg2_1.view(ROWS, FEATURES)
    mask1_flat = arg3_1.view(ROWS, FEATURES)
    out_f32_flat = out_f32.view(ROWS, FEATURES)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (row_tiles, FEATURES // BLOCK_FEATURES, 1),
        _materialize_partial_kernel,
        (arg0_1, arg1_1, mask0_flat, mask1_flat, out_f32_flat, out_bf16, partial,
         ROW_BLOCK, BLOCK_FEATURES, SEQ),
    )
    ct.launch(
        stream,
        (FEATURES // BLOCK_FEATURES, 1, 1),
        _finish_sum_kernel,
        (partial, sum_out, row_tiles, BLOCK_FEATURES),
    )
    return out_f32, out_bf16, out_bf16.t(), sum_out
