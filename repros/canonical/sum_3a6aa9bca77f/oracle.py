"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 MobileViT SiLU-gradient producer, returned contiguous `[ROWS, FEATURES]` bf16 tensor, its `[FEATURES, ROWS]` transpose alias, and the bf16-rounded fp32 feature sum by materializing each producer tile once and accumulating split-row partials from the same bf16-rounded values, whereas Inductor schedules the sigmoid/pointwise materialization, alias returns, and dim-0 sum as separate generic work over the materialized producer; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K template that preserves a live returned bf16 side output, metadata-only transpose alias, the generated `tl.sigmoid` arithmetic, and the bf16 cast boundary before a sibling column reduction; the fix is COOPERATIVE_SPLIT_K: add a multi-output row-split reduction schedule that writes the returned tensor once, forms alias-only view returns, and finalizes compatible feature sums from shared partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _silu_grad_producer(lhs_bf16, rhs_bf16):
    lhs = lhs_bf16.to(tl.float32)
    rhs = rhs_bf16.to(tl.float32)
    sigmoid = tl.sigmoid(rhs)
    first = lhs * sigmoid
    tail = rhs * (1.0 - sigmoid) + 1.0
    return first * tail


@triton.jit
def _materialize_partial_sum_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    partial_ptr,
    ROWS: tl.constexpr,
    FEATURES: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    group = tl.program_id(0)
    feature_block = tl.program_id(1)
    features = feature_block * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    feature_mask = features < FEATURES
    acc = tl.zeros((BLOCK_FEATURES,), dtype=tl.float32)

    for start in tl.range(0, GROUP_ROWS, BLOCK_ROWS):
        rows = group * GROUP_ROWS + start + tl.arange(0, BLOCK_ROWS)
        offsets = rows[:, None] * FEATURES + features[None, :]
        mask = (rows[:, None] < ROWS) & feature_mask[None, :]

        lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
        value_bf16 = _silu_grad_producer(lhs, rhs).to(tl.bfloat16)

        tl.store(out_ptr + offsets, value_bf16, mask=mask)
        acc += tl.sum(tl.where(mask, value_bf16.to(tl.float32), 0.0), axis=0)

    tl.store(
        partial_ptr + group * FEATURES + features,
        acc,
        mask=feature_mask,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_GROUPS: tl.constexpr,
    FEATURES: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    feature_block = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    features = feature_block * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    mask = (groups[:, None] < NUM_GROUPS) & (features[None, :] < FEATURES)
    values = tl.load(
        partial_ptr + groups[:, None] * FEATURES + features[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + features, rounded, mask=features < FEATURES)


# 57ead6f2: (T([131072,288], bf16), T([131072,288], bf16), ...)
@oracle_impl(hardware="B200", point="57ead6f2", GROUP_ROWS=256, BLOCK_ROWS=256, BLOCK_FEATURES=32, FINAL_BLOCK_FEATURES=32, num_warps=8)
# 459ab323: (T([32768,384], bf16), T([32768,384], bf16), ...)
@oracle_impl(hardware="B200", point="459ab323", GROUP_ROWS=512, BLOCK_ROWS=64, BLOCK_FEATURES=32, FINAL_BLOCK_FEATURES=32, num_warps=8)
# f681e57a: (T([8192,480], bf16), T([8192,480], bf16), ...)
@oracle_impl(hardware="B200", point="f681e57a", GROUP_ROWS=256, BLOCK_ROWS=64, BLOCK_FEATURES=32, FINAL_BLOCK_FEATURES=32, num_warps=8)
def oracle_forward(
    inputs,
    *,
    GROUP_ROWS: int,
    BLOCK_ROWS: int,
    BLOCK_FEATURES: int,
    FINAL_BLOCK_FEATURES: int,
    num_warps: int,
):
    lhs, rhs, _shape0, _shape1, flat_shape_arg, sum_shape_arg = inputs
    flat_shape = tuple(int(dim) for dim in flat_shape_arg)
    sum_shape = tuple(int(dim) for dim in sum_shape_arg)
    rows = int(flat_shape[0])
    features = int(flat_shape[1])

    out = torch.empty_strided(
        flat_shape,
        (features, 1),
        device=lhs.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        sum_shape,
        (1,),
        device=lhs.device,
        dtype=torch.float32,
    )
    num_groups = triton.cdiv(rows, GROUP_ROWS)
    partial = torch.empty_strided(
        (num_groups, features),
        (features, 1),
        device=lhs.device,
        dtype=torch.float32,
    )

    _materialize_partial_sum_kernel[(num_groups, triton.cdiv(features, BLOCK_FEATURES))](
        lhs,
        rhs,
        out,
        partial,
        ROWS=rows,
        FEATURES=features,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_FEATURES=BLOCK_FEATURES,
        num_warps=num_warps,
        num_stages=4,
    )
    _final_sum_kernel[(triton.cdiv(features, FINAL_BLOCK_FEATURES),)](
        partial,
        sum_out,
        NUM_GROUPS=num_groups,
        FEATURES=features,
        BLOCK_GROUPS=triton.next_power_of_2(num_groups),
        BLOCK_FEATURES=FINAL_BLOCK_FEATURES,
        num_warps=num_warps,
        num_stages=1,
    )

    transposed = torch.as_strided(out, (features, rows), (1, features))
    return out, transposed, sum_out
