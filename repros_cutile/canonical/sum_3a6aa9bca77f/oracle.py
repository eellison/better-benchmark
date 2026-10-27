"""cuTile port of sum_3a6aa9bca77f (COOPERATIVE_SPLIT_K): MobileViT SiLU-gradient
producer + returned bf16 tensor + bf16-rounded feature sum.

Matches Triton's split-K: BLOCK_ROWS x BLOCK_FEATURES tiles walk GROUP_ROWS
per block, accumulating a partial column sum, then a final reduce.
Uses padding_mode=ZERO to avoid full-tensor pad copies.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _materialize_partial_sum_kernel(
    lhs_ptr,       # bf16 [ROWS, FEATURES]
    rhs_ptr,       # bf16 [ROWS, FEATURES]
    out_ptr,       # bf16 [ROWS, FEATURES]
    partial_ptr,   # f32 [NUM_GROUPS, FEATURES]  (padded features)
    ROWS: ct.Constant[int],
    FEATURES: ct.Constant[int],
    GROUP_ROWS: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_FEATURES: ct.Constant[int],
    NUM_ROW_STEPS: ct.Constant[int],
):
    group = ct.bid(0)
    feature_block = ct.bid(1)

    acc = ct.zeros((BLOCK_FEATURES,), dtype=ct.float32)
    for step in range(NUM_ROW_STEPS):
        row_start = group * (GROUP_ROWS // BLOCK_ROWS) + step
        lhs_bf = ct.load(lhs_ptr, index=(row_start, feature_block),
                         shape=(BLOCK_ROWS, BLOCK_FEATURES),
                         padding_mode=ct.PaddingMode.ZERO)
        rhs_bf = ct.load(rhs_ptr, index=(row_start, feature_block),
                         shape=(BLOCK_ROWS, BLOCK_FEATURES),
                         padding_mode=ct.PaddingMode.ZERO)
        lhs = ct.astype(lhs_bf, ct.float32)
        rhs = ct.astype(rhs_bf, ct.float32)
        sigmoid = 1.0 / (1.0 + ct.exp(-rhs))
        value = lhs * sigmoid * (rhs * (1.0 - sigmoid) + 1.0)
        value_bf = ct.astype(value, ct.bfloat16)
        # Store (may write junk to OOB rows/cols — but we allocated 'out'
        # to be exactly [ROWS, FEATURES], so OOB writes are actually into
        # legal memory only if we clamp; use mask via scatter).
        ct.store(out_ptr, index=(row_start, feature_block), tile=value_bf)
        # Accumulate into column partial
        v_f = ct.astype(value_bf, ct.float32)
        acc = acc + ct.sum(v_f, axis=0)

    ct.store(partial_ptr, index=(group, feature_block),
             tile=ct.reshape(acc, (1, BLOCK_FEATURES)))


@ct.kernel
def _final_sum_kernel(
    partial_ptr,   # f32 [NUM_GROUPS, FEATURES]
    sum_ptr,       # f32 [FEATURES]
    NUM_GROUPS: ct.Constant[int],
    FEATURES: ct.Constant[int],
    BLOCK_FEATURES: ct.Constant[int],
):
    feat_block = ct.bid(0)
    vals = ct.load(partial_ptr, index=(0, feat_block),
                   shape=(NUM_GROUPS, BLOCK_FEATURES))
    total = ct.sum(vals, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(feat_block,), tile=rounded)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="57ead6f2",
             GROUP_ROWS=256, BLOCK_ROWS=256, BLOCK_FEATURES=32,
             FINAL_BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="459ab323",
             GROUP_ROWS=512, BLOCK_ROWS=64, BLOCK_FEATURES=32,
             FINAL_BLOCK_FEATURES=32)
@oracle_impl(hardware="B200", point="f681e57a",
             GROUP_ROWS=256, BLOCK_ROWS=64, BLOCK_FEATURES=32,
             FINAL_BLOCK_FEATURES=32)
def oracle_forward(inputs, *, GROUP_ROWS, BLOCK_ROWS, BLOCK_FEATURES,
                   FINAL_BLOCK_FEATURES):
    lhs, rhs, _shape0, _shape1, flat_shape_arg, sum_shape_arg = inputs
    flat_shape = _shape_tuple(flat_shape_arg)
    sum_shape = _shape_tuple(sum_shape_arg)
    rows = int(flat_shape[0])
    features = int(flat_shape[1])
    device = lhs.device

    out = torch.empty_strided(
        flat_shape, (features, 1), device=device, dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)

    num_groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    feat_blocks = (features + BLOCK_FEATURES - 1) // BLOCK_FEATURES
    partial = torch.zeros(
        (num_groups, feat_blocks * BLOCK_FEATURES),
        device=device, dtype=torch.float32,
    )
    num_row_steps = GROUP_ROWS // BLOCK_ROWS

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, feat_blocks, 1),
        _materialize_partial_sum_kernel,
        (lhs, rhs, out, partial, rows, features, GROUP_ROWS, BLOCK_ROWS,
         BLOCK_FEATURES, num_row_steps),
    )

    # Final reduce: features may be non-power-of-2, and NUM_GROUPS must be pow2.
    num_groups_pow2 = _next_pow2(num_groups)
    if num_groups_pow2 != num_groups:
        # Pad rows so NUM_GROUPS is power of 2 for cuTile tile load.
        partial_pow2 = torch.zeros(
            (num_groups_pow2, feat_blocks * BLOCK_FEATURES),
            device=device, dtype=torch.float32,
        )
        partial_pow2[:num_groups] = partial
        partial = partial_pow2

    sum_padded = torch.empty(
        (feat_blocks * FINAL_BLOCK_FEATURES,), device=device, dtype=torch.float32
    )
    ct.launch(
        stream,
        (feat_blocks, 1, 1),
        _final_sum_kernel,
        (partial, sum_padded, num_groups_pow2, features, FINAL_BLOCK_FEATURES),
    )

    sum_out.copy_(sum_padded[:features])

    transposed = torch.as_strided(out, (features, rows), (1, features))
    return out, transposed, sum_out
