"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DeBERTa attention-output divide, head reorder clone, returned contiguous base and transpose alias, and sibling hidden-dimension sum by writing the final contiguous [4096, 1536] backing storage while accumulating bf16-rounded column-sum partials from the same tile stream, whereas Inductor schedules the div/view/permute/clone/view materialization and dim-0 sum as separate generic layout and reduction work over the materialized clone; Inductor cannot do this today because its scheduler does not preserve a layout-changing clone producer as a multi-output reduction source that can both return the materialized layout aliases and emit compatible column-reduction partials with the captured bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a layout-materializing multi-output reduction schedule for fixed attention head reorders with sibling hidden-dimension reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
HEADS = 24
QUERY = 64
KEY = 512
ROWS = BATCH * KEY
FEATURES = HEADS * QUERY
ROW_BLOCK = 64
COL_BLOCK = 64
NUM_ROW_BLOCKS = triton.cdiv(ROWS, ROW_BLOCK)


@triton.jit
def _layout_div_reduce_partials_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    partial_ptr,
    x_stride_bh: tl.constexpr,
    x_stride_q: tl.constexpr,
    x_stride_k: tl.constexpr,
    HEADS_: tl.constexpr,
    QUERY_: tl.constexpr,
    KEY_: tl.constexpr,
    ROWS_: tl.constexpr,
    FEATURES_: tl.constexpr,
    ROW_BLOCK_: tl.constexpr,
    COL_BLOCK_: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * ROW_BLOCK_ + tl.arange(0, ROW_BLOCK_)
    cols = col_block * COL_BLOCK_ + tl.arange(0, COL_BLOCK_)

    batch = rows // KEY_
    key = rows - batch * KEY_
    head = cols // QUERY_
    query = cols - head * QUERY_
    batch_head = batch[:, None] * HEADS_ + head[None, :]

    mask = (rows[:, None] < ROWS_) & (cols[None, :] < FEATURES_)
    input_offsets = (
        batch_head * x_stride_bh
        + query[None, :] * x_stride_q
        + key[:, None] * x_stride_k
    )
    out_offsets = rows[:, None] * FEATURES_ + cols[None, :]

    scale = tl.load(scale_ptr).to(tl.float32)
    values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    div_bf16 = (values / scale).to(tl.bfloat16)
    div_f32 = div_bf16.to(tl.float32)

    tl.store(out_ptr + out_offsets, div_bf16, mask=mask)
    partial = tl.sum(tl.where(mask, div_f32, 0.0), axis=0)
    tl.store(partial_ptr + row_block * FEATURES_ + cols, partial, mask=cols < FEATURES_)


@triton.jit
def _finish_sum_kernel(
    partial_ptr,
    sum_ptr,
    FEATURES_: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    col = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)
    partials = tl.load(partial_ptr + row_blocks * FEATURES_ + col).to(tl.float32)
    rounded = tl.sum(partials, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + col, rounded)


# (T([192,64,512], bf16), T([], bf16), S([8,24,512,64]), S([8,512,1536]), S([4096,1536]), S([1536]))
@oracle_impl(hardware="B200", point="9929f4a9", num_warps=8)
def oracle_forward(inputs, *, num_warps: int):
    x, scale, _shape0, _shape1, shape2, _shape3 = inputs
    out_base = torch.empty(
        tuple(int(dim) for dim in shape2),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (NUM_ROW_BLOCKS, FEATURES),
        device=x.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty((FEATURES,), device=x.device, dtype=torch.float32)

    grid = (NUM_ROW_BLOCKS, triton.cdiv(FEATURES, COL_BLOCK))
    _layout_div_reduce_partials_kernel[grid](
        x,
        scale,
        out_base,
        partials,
        x_stride_bh=x.stride(0),
        x_stride_q=x.stride(1),
        x_stride_k=x.stride(2),
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
        ROWS_=ROWS,
        FEATURES_=FEATURES,
        ROW_BLOCK_=ROW_BLOCK,
        COL_BLOCK_=COL_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _finish_sum_kernel[(FEATURES,)](
        partials,
        out_sum,
        FEATURES_=FEATURES,
        BLOCK_R=NUM_ROW_BLOCKS,
        num_warps=8,
        num_stages=1,
    )
    out_transpose = torch.as_strided(out_base, (FEATURES, ROWS), (1, FEATURES))
    return out_base, out_transpose, out_sum
