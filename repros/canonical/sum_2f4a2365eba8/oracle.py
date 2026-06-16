"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the attention-output head reorder clone with the sibling hidden-dimension sum, writing the returned contiguous `[ROWS, FEATURES]` bf16 buffer, returning its `[FEATURES, ROWS]` transpose view, and accumulating the fp32 column sum with the required bf16 round-trip before the final fp32 output, whereas Inductor currently schedules the layout-changing clone, returned views, and dim-0 reduction as separate generic work over the materialized clone; Inductor cannot do this today because its scheduler does not preserve a layout-materializing clone producer as a multi-output reduction source that can both return aliased layout views and emit compatible column-reduction partials with the captured cast boundary; the fix is SCHEDULER_FUSION: add a layout-materializing multi-output reduction schedule for fixed attention head reorders with sibling hidden-dimension reductions and dtype-rounding epilogues."""

import torch
import triton
import triton.language as tl
from oracle_harness import oracle_impl


@triton.jit
def _copy_reduce_kernel(
    x_ptr,
    clone_ptr,
    partial_ptr,
    STRIDE_BH: tl.constexpr,
    STRIDE_S: tl.constexpr,
    STRIDE_D: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQ: tl.constexpr,
    ROWS: tl.constexpr,
    FEATURES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    row_block = tl.program_id(0)
    feature_block = tl.program_id(1)

    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    features = feature_block * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    mask = (rows[:, None] < ROWS) & (features[None, :] < FEATURES)

    batch = rows // SEQ
    seq = rows - batch * SEQ
    head = features // HEAD_DIM
    dim = features - head * HEAD_DIM
    bh = batch[:, None] * HEADS + head[None, :]

    x_offsets = bh * STRIDE_BH + seq[:, None] * STRIDE_S + dim[None, :] * STRIDE_D
    out_offsets = rows[:, None] * FEATURES + features[None, :]
    values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(clone_ptr + out_offsets, values, mask=mask)
    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(
        partial_ptr + row_block * FEATURES + features,
        partial,
        mask=features < FEATURES,
    )


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    FEATURES: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
):
    feature_block = tl.program_id(0)
    features = feature_block * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (features[None, :] < FEATURES)
    offsets = row_blocks[:, None] * FEATURES + features[None, :]
    values = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(values, axis=0)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + features, rounded, mask=features < FEATURES)


def _shape(value):
    return tuple(int(dim) for dim in value)


@oracle_impl(hardware="B200", point="fb089404", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="07e248d7", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="576ca76e", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="471d82af", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="c6b0f684", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="2cdbce9d", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="c23ba4e7", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="a3cab238", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="14c0be85", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
@oracle_impl(hardware="B200", point="d528e08b", BLOCK_ROWS=64, BLOCK_FEATURES=32, num_warps=8)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_FEATURES, num_warps):
    x, shape0, _shape1, shape2, _shape3 = inputs
    batch, heads, seq, head_dim = _shape(shape0)
    rows, features = _shape(shape2)

    clone = torch.empty_strided(
        (rows, features),
        (features, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_sum = torch.empty((features,), device=x.device, dtype=torch.float32)
    num_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
    partials = torch.empty(
        (num_row_blocks, features),
        device=x.device,
        dtype=torch.float32,
    )

    grid = (num_row_blocks, triton.cdiv(features, BLOCK_FEATURES))
    _copy_reduce_kernel[grid](
        x,
        clone,
        partials,
        STRIDE_BH=x.stride(0),
        STRIDE_S=x.stride(1),
        STRIDE_D=x.stride(2),
        HEADS=heads,
        HEAD_DIM=head_dim,
        SEQ=seq,
        ROWS=rows,
        FEATURES=features,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_FEATURES=BLOCK_FEATURES,
        num_warps=num_warps,
        num_stages=4,
    )
    _finish_sum_kernel[(triton.cdiv(features, BLOCK_FEATURES),)](
        partials,
        out_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        FEATURES=features,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_FEATURES=BLOCK_FEATURES,
        num_warps=num_warps,
        num_stages=1,
    )

    transposed = torch.as_strided(clone, (features, rows), (1, features))
    return clone, transposed, out_sum
