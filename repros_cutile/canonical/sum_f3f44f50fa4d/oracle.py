"""cuTile port of sum_f3f44f50fa4d: scaled attention-head clone + column sum.

Ports the Triton `_scaled_copy_reduce_kernel` + `_finish_sum_kernel` pattern:
compute the scaled bf16 clone and per-row-tile column partials in one kernel,
then reduce partials over row tiles in a small finalizer.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.334370152488211
BLOCK_ROWS = 64
BLOCK_FEATURES = 64


@ct.kernel
def _scaled_copy_reduce_kernel(
    x_ptr,           # bf16 flat, strided arg0_1 [B, H, D, S]
    clone_ptr,       # bf16 flat contiguous [ROWS, FEATURES]
    partial_ptr,     # f32 [NUM_ROW_TILES, FEATURES]
    STRIDE_BH: ct.Constant[int],
    STRIDE_D: ct.Constant[int],
    STRIDE_S: ct.Constant[int],
    HEADS: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    SEQ: ct.Constant[int],
    ROWS: ct.Constant[int],
    FEATURES: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_ROWS_C: ct.Constant[int],
    BLOCK_FEATURES_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    feature_block = ct.bid(1)

    rows = row_block * BLOCK_ROWS_C + ct.arange(BLOCK_ROWS_C, dtype=ct.int32)
    features = feature_block * BLOCK_FEATURES_C + ct.arange(BLOCK_FEATURES_C, dtype=ct.int32)

    rows_2d = ct.reshape(rows, (BLOCK_ROWS_C, 1))
    features_2d = ct.reshape(features, (1, BLOCK_FEATURES_C))
    mask = (rows_2d < ROWS) & (features_2d < FEATURES)

    batch = rows_2d // SEQ
    seq = rows_2d - batch * SEQ
    head = features_2d // HEAD_DIM
    dim = features_2d - head * HEAD_DIM
    bh = batch * HEADS + head

    x_offsets = bh * STRIDE_BH + dim * STRIDE_D + seq * STRIDE_S
    out_offsets = rows_2d * FEATURES + features_2d

    values_bf = ct.gather(x_ptr, x_offsets)
    scaled = ct.astype(values_bf, ct.float32) * SCALE_
    values = ct.astype(scaled, ct.bfloat16)

    ct.scatter(clone_ptr, out_offsets, values, mask=mask)

    values_f = ct.where(mask, ct.astype(values, ct.float32),
                        ct.zeros((BLOCK_ROWS_C, BLOCK_FEATURES_C), dtype=ct.float32))
    partial = ct.sum(values_f, axis=0, keepdims=False)
    p_offsets = row_block * FEATURES + features
    p_mask = features < FEATURES
    ct.scatter(partial_ptr, p_offsets, partial, mask=p_mask)


@ct.kernel
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_BLOCKS: ct.Constant[int],
    FEATURES: ct.Constant[int],
    BLOCK_ROW_BLOCKS: ct.Constant[int],
    BLOCK_FEATURES_C: ct.Constant[int],
):
    feature_block = ct.bid(0)
    features = feature_block * BLOCK_FEATURES_C + ct.arange(BLOCK_FEATURES_C, dtype=ct.int32)
    row_blocks = ct.arange(BLOCK_ROW_BLOCKS, dtype=ct.int32)
    rb_2d = ct.reshape(row_blocks, (BLOCK_ROW_BLOCKS, 1))
    f_2d = ct.reshape(features, (1, BLOCK_FEATURES_C))
    mask = (rb_2d < NUM_ROW_BLOCKS) & (f_2d < FEATURES)
    offsets = rb_2d * FEATURES + f_2d
    values_bf = ct.gather(partial_ptr, offsets)
    values = ct.astype(values_bf, ct.float32)
    values = ct.where(mask, values,
                      ct.zeros((BLOCK_ROW_BLOCKS, BLOCK_FEATURES_C), dtype=ct.float32))
    total = ct.sum(values, axis=0, keepdims=False)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.scatter(sum_ptr, features, rounded, mask=features < FEATURES)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="115f9a0f")
@oracle_impl(hardware="B200", point="471fe712")
@oracle_impl(hardware="B200", point="17865e49")
@oracle_impl(hardware="B200", point="144bae60")
@oracle_impl(hardware="B200", point="34b9f4a9")
@oracle_impl(hardware="B200", point="5f1be6f8")
def oracle_forward(inputs):
    x, shape0, _shape1, shape2, _shape3 = inputs
    _batch, heads, head_dim, seq = (int(d) for d in shape0)
    rows, features = (int(d) for d in shape2)

    clone = torch.empty_strided(
        (rows, features), (features, 1),
        device=x.device, dtype=torch.bfloat16,
    )
    out_sum = torch.empty((features,), device=x.device, dtype=torch.float32)
    num_row_tiles = (rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    num_feat_tiles = (features + BLOCK_FEATURES - 1) // BLOCK_FEATURES
    partials = torch.empty(
        (num_row_tiles, features), device=x.device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_tiles, num_feat_tiles, 1),
        _scaled_copy_reduce_kernel,
        (x.view(-1), clone.view(-1), partials.view(-1),
         int(x.stride(0)), int(x.stride(1)), int(x.stride(2)),
         heads, head_dim, seq, rows, features, SCALE,
         BLOCK_ROWS, BLOCK_FEATURES),
    )
    block_row_blocks = _next_pow2(num_row_tiles)
    ct.launch(
        stream, (num_feat_tiles, 1, 1),
        _finish_sum_kernel,
        (partials.view(-1), out_sum, num_row_tiles, features,
         block_row_blocks, BLOCK_FEATURES),
    )
    permute_1 = clone.permute(1, 0)
    return clone, permute_1, out_sum
