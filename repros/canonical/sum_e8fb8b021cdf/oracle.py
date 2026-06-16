"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 scaled attention-head layout materialization, returned contiguous view, returned transposed alias, and fp32 column sum rounded through bf16 from one fused materialize-and-partial-reduce pass plus a small finalizer, whereas Inductor schedules the captured view/permute/clone/mul/view/transpose/sum/convert chain through generic layout, pointwise, and reduction regions; Inductor cannot do this today because its scheduler does not keep a scalar-producing layout clone live as a multi-output reduction source that both returns aliasing views and feeds a sibling rounded column reduction; the fix is SCHEDULER_FUSION: add a guarded scaled head-layout multi-output reduction template that sinks the scalar multiply into the materialization, returns the aliasing views, and accumulates the reduction across the captured bf16 and fp32 cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scaled_copy_reduce_kernel(
    x_ptr,
    out_ptr,
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
    values = tl.load(
        x_ptr + x_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    scaled = (values * 0.125).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + out_offsets, scaled, mask=mask)
    partial = tl.sum(tl.where(mask, scaled.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + row_block * FEATURES + features,
        partial,
        mask=features < FEATURES,
    )


@triton.jit
def _final_sum_kernel(
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
    values = tl.load(
        partial_ptr + row_blocks[:, None] * FEATURES + features[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + features, rounded, mask=features < FEATURES)


def _shape(value):
    return tuple(int(dim) for dim in value)


# a3cab238: (T([1024,256,64], bf16), S([64,16,256,64]), S([64,256,1024]), S([16384,1024]), S([1024]))
@oracle_impl(hardware="B200", point="a3cab238", BLOCK_ROWS=128, BLOCK_FEATURES=32, num_warps=8)
# 14c0be85: (T([512,128,64], bf16), S([32,16,128,64]), S([32,128,1024]), S([4096,1024]), S([1024]))
@oracle_impl(hardware="B200", point="14c0be85", BLOCK_ROWS=128, BLOCK_FEATURES=32, num_warps=8)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_FEATURES, num_warps):
    x, shape0, _shape1, shape2, shape3 = inputs
    _batch, heads, seq, head_dim = _shape(shape0)
    rows, features = _shape(shape2)

    scaled = torch.empty_strided(
        (rows, features),
        (features, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    num_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
    partial = torch.empty(
        (num_row_blocks, features),
        device=x.device,
        dtype=torch.float32,
    )
    reduced = torch.empty_strided(
        _shape(shape3),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    grid = (num_row_blocks, triton.cdiv(features, BLOCK_FEATURES))
    _scaled_copy_reduce_kernel[grid](
        x,
        scaled,
        partial,
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
    _final_sum_kernel[(triton.cdiv(features, BLOCK_FEATURES),)](
        partial,
        reduced,
        NUM_ROW_BLOCKS=num_row_blocks,
        FEATURES=features,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_FEATURES=BLOCK_FEATURES,
        num_warps=8,
        num_stages=1,
    )
    return scaled, scaled.permute(1, 0), reduced
