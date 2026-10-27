"""cuTile port of sum_e8fb8b021cdf (SCHEDULER_FUSION): scaled head-layout
materialization plus dim-0 column sum + returned permute alias.

Input `arg0_1[BH, S, D]` is reshaped to `[batch, heads, seq, D]`, permuted to
`[batch, seq, heads, D]`, viewed as `[rows, features]` = `[batch*seq, heads*D]`,
and multiplied by 0.125 (bf16-rounded). The oracle also returns the column
sum via a two-pass partial+finalize.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scaled_copy_reduce_kernel(
    x_permuted_ptr,  # bf16 [rows, features] (already permuted)
    out_ptr,         # bf16 [rows, features]
    partial_ptr,     # f32 [NUM_ROW_BLOCKS, features]
    ROWS: ct.Constant[int],
    FEATURES: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_FEATURES: ct.Constant[int],
):
    row_block = ct.bid(0)
    feature_block = ct.bid(1)
    values = ct.load(x_permuted_ptr, index=(row_block, feature_block), shape=(BLOCK_ROWS, BLOCK_FEATURES))
    values_f = ct.astype(values, ct.float32)
    scaled_bf16 = ct.astype(values_f * 0.125, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, feature_block), tile=scaled_bf16)

    partial = ct.sum(ct.astype(scaled_bf16, ct.float32), axis=0)
    ct.store(
        partial_ptr,
        index=(row_block, feature_block),
        tile=ct.reshape(partial, (1, BLOCK_FEATURES)),
    )


@ct.kernel
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_FEATURES: ct.Constant[int],
):
    feature_block = ct.bid(0)
    partials = ct.load(
        partial_ptr, index=(0, feature_block), shape=(NUM_ROW_BLOCKS, BLOCK_FEATURES)
    )
    total = ct.sum(ct.astype(partials, ct.float32), axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(feature_block,), tile=rounded)


def _shape(value):
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="B200", point="a3cab238", BLOCK_ROWS=128, BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="14c0be85", BLOCK_ROWS=128, BLOCK_FEATURES=32)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_FEATURES):
    x, shape0, _shape1, shape2, shape3 = inputs
    _batch, heads, seq, head_dim = _shape(shape0)
    rows, features = _shape(shape2)

    # x is bf16[bh, s, d] = view of [batch*heads, seq, head_dim].
    # We need to view as [batch, heads, seq, head_dim], permute to
    # [batch, seq, heads, head_dim], and contiguous-copy for cuTile.
    x_reshaped = x.view(_batch, heads, seq, head_dim)
    x_permuted = x_reshaped.permute(0, 2, 1, 3).contiguous().view(rows, features)

    scaled = torch.empty_strided(
        (rows, features),
        (features, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = (rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    partial = torch.empty((num_row_blocks, features), device=x.device, dtype=torch.float32)
    reduced = torch.empty_strided(
        _shape(shape3), (1,), device=x.device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_blocks, (features + BLOCK_FEATURES - 1) // BLOCK_FEATURES, 1),
        _scaled_copy_reduce_kernel,
        (x_permuted, scaled, partial, rows, features, BLOCK_ROWS, BLOCK_FEATURES),
    )
    ct.launch(
        stream,
        ((features + BLOCK_FEATURES - 1) // BLOCK_FEATURES, 1, 1),
        _final_sum_kernel,
        (partial, reduced, num_row_blocks, BLOCK_FEATURES),
    )
    return scaled, scaled.permute(1, 0), reduced
