"""cuTile port of sum_76fb9946e296: attention head-reorder + column sum.

Reads bf16 source `[ROWS, HEAD_DIM]` (through a strided head-major view) with
shape (B*H, S, D), scales by SCALE, writes contiguous bf16 clone `[B*S, H*D]`
and returns aliased transpose + fp32 column sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _copy_reduce_kernel(
    x_view,       # bf16 (B*S, H, D)   -- reshaped view of x = (B*H, S, D) permuted
    clone_ptr,    # bf16 (ROWS, FEATURES)
    partial_ptr,  # f32 (NUM_ROW_BLOCKS, FEATURES)
    HEADS: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    ROWS_C: ct.Constant[int],
    FEATURES_C: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_FEATURES: ct.Constant[int],
    SCALE_SCAL: ct.Constant[float],
):
    row_block = ct.bid(0)
    feature_block = ct.bid(1)

    # Load a (BLOCK_ROWS, HEADS, HEAD_DIM) slice? We need direct 2D indexing.
    # x_view is (ROWS, FEATURES) after remapping. Just load & scale.
    values = ct.load(x_view, index=(row_block, feature_block), shape=(BLOCK_ROWS, BLOCK_FEATURES))
    values_f = ct.astype(values, ct.float32)
    scaled = values_f * SCALE_SCAL
    rounded = ct.astype(scaled, ct.bfloat16)
    ct.store(clone_ptr, index=(row_block, feature_block), tile=rounded)

    partial = ct.sum(ct.astype(rounded, ct.float32), axis=0)  # (BLOCK_FEATURES,)
    partial_2d = ct.reshape(partial, (1, BLOCK_FEATURES))
    ct.store(partial_ptr, index=(row_block, feature_block), tile=partial_2d)


@ct.kernel
def _finish_sum_kernel(
    partial_ptr,     # f32 (NUM_ROW_BLOCKS_POW2, FEATURES)
    sum_ptr,         # f32 (FEATURES,)
    NUM_ROW_BLOCKS_C: ct.Constant[int],
    FEATURES_C: ct.Constant[int],
    BLOCK_ROW_BLOCKS_POW2: ct.Constant[int],
    BLOCK_FEATURES: ct.Constant[int],
):
    feature_block = ct.bid(0)
    partials = ct.load(partial_ptr, index=(0, feature_block), shape=(BLOCK_ROW_BLOCKS_POW2, BLOCK_FEATURES))
    total = ct.sum(partials, axis=0)  # (BLOCK_FEATURES,)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(feature_block,), tile=rounded)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="07e248d7", BLOCK_ROWS=128, BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="471d82af", BLOCK_ROWS=64, BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="c6b0f684", BLOCK_ROWS=64, BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="2cdbce9d", BLOCK_ROWS=64, BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="c23ba4e7", BLOCK_ROWS=64, BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="d528e08b", BLOCK_ROWS=64, BLOCK_FEATURES=32)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_FEATURES):
    x, shape0, _shape1, shape2, _shape3 = inputs
    _batch, heads, seq, head_dim = (int(dim) for dim in shape0)
    rows, features = (int(dim) for dim in shape2)

    # x is (B*H, S, D) with stride possibly not contiguous. We need to construct a
    # (rows, features) = (B*S, H*D) view that reorders heads out.
    # x.shape = (B*H, S, D) — reshape to (B, H, S, D), permute (0,2,1,3), reshape.
    batch = _batch
    x_4d = x.view(batch, heads, seq, head_dim)
    x_reordered = x_4d.permute(0, 2, 1, 3).contiguous().view(rows, features)

    clone = torch.empty_strided(
        (rows, features), (features, 1),
        device=x.device, dtype=torch.bfloat16,
    )
    num_row_blocks = (rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    num_row_blocks_pow2 = _next_pow2(num_row_blocks)
    partials = torch.zeros((num_row_blocks_pow2, features), device=x.device, dtype=torch.float32)
    out_sum = torch.empty((features,), device=x.device, dtype=torch.float32)

    SCALE = 0.334370152488211
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_blocks, (features + BLOCK_FEATURES - 1) // BLOCK_FEATURES, 1),
        _copy_reduce_kernel,
        (x_reordered, clone, partials,
         heads, head_dim, rows, features,
         BLOCK_ROWS, BLOCK_FEATURES, SCALE),
    )
    ct.launch(
        stream,
        ((features + BLOCK_FEATURES - 1) // BLOCK_FEATURES, 1, 1),
        _finish_sum_kernel,
        (partials, out_sum,
         num_row_blocks, features, num_row_blocks_pow2, BLOCK_FEATURES),
    )

    transposed = torch.as_strided(clone, (features, rows), (1, features))
    return clone, transposed, out_sum
