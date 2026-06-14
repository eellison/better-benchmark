"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full SigLIP two-input QK cat/view/permute/clone layout by writing the returned contiguous `[32768,1536]` bf16 backing tensor, returning its `[1536,32768]` transpose alias, and accumulating the sibling column-sum partials from the same source traversal before applying the captured bf16 sum-result cast; Inductor materializes the clone and then rereads it through a separate reduction, because scheduler/codegen does not fuse a mandatory layout-changing returned producer with a reduction consumer over the same logical buffer while preserving aliasing and dtype boundaries; the fix is SCHEDULER_FUSION: teach layout materialization to emit same-tile partial reductions for returned clone/view producers with simple column-sum consumers."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
HEADS = 12
TOKENS = 256
HEAD_DIM = 64
SOURCES = 2
ROWS = BATCH * TOKENS
SOURCE_COLS = HEADS * HEAD_DIM
FEATURES = SOURCES * SOURCE_COLS
INPUT_ROW_STRIDE = 196608
INPUT_HEAD_STRIDE = 64
INPUT_TOKEN_STRIDE = 768
INPUT_DIM_STRIDE = 1
CHUNK_ROWS = 373
NUM_CHUNKS = triton.cdiv(ROWS, CHUNK_ROWS)
ROW_BLOCK = 32
FEATURE_BLOCK = 64
FINAL_CHUNK_BLOCK = 128


@triton.jit
def _layout_partial_sum_kernel(
    in0_ptr,
    in1_ptr,
    layout_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    FEATURES_: tl.constexpr,
    TOKENS_: tl.constexpr,
    INPUT_ROW_STRIDE_: tl.constexpr,
    INPUT_HEAD_STRIDE_: tl.constexpr,
    INPUT_TOKEN_STRIDE_: tl.constexpr,
    INPUT_DIM_STRIDE_: tl.constexpr,
    SOURCE_COLS_: tl.constexpr,
    CHUNK_ROWS_: tl.constexpr,
    ROW_BLOCK_: tl.constexpr,
    FEATURE_BLOCK_: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    feature_block = tl.program_id(1)
    features = feature_block * FEATURE_BLOCK_ + tl.arange(0, FEATURE_BLOCK_)
    feature_mask = features < FEATURES_
    source = features // SOURCE_COLS_
    source_feature = features - source * SOURCE_COLS_
    head = source_feature // 64
    dim = source_feature - head * 64
    acc = tl.zeros((FEATURE_BLOCK_,), dtype=tl.float32)

    for row_start in tl.range(0, CHUNK_ROWS_, ROW_BLOCK_):
        local_rows = row_start + tl.arange(0, ROW_BLOCK_)
        rows = chunk_id * CHUNK_ROWS_ + local_rows
        row_mask = (local_rows < CHUNK_ROWS_) & (rows < ROWS_)
        batch = rows // TOKENS_
        token = rows - batch * TOKENS_
        offsets = (
            batch[:, None] * INPUT_ROW_STRIDE_
            + head[None, :] * INPUT_HEAD_STRIDE_
            + token[:, None] * INPUT_TOKEN_STRIDE_
            + dim[None, :] * INPUT_DIM_STRIDE_
        )
        mask = row_mask[:, None] & feature_mask[None, :]

        values0 = tl.load(in0_ptr + offsets, mask=mask & (source[None, :] == 0), other=0.0)
        values1 = tl.load(in1_ptr + offsets, mask=mask & (source[None, :] == 1), other=0.0)
        values = values0 + values1
        layout_offsets = rows[:, None] * FEATURES_ + features[None, :]
        tl.store(layout_ptr + layout_offsets, values, mask=mask)
        acc += tl.sum(tl.where(mask, values.to(tl.float32), 0.0), axis=0)

    tl.store(
        partial_ptr + chunk_id * FEATURES_ + features,
        acc,
        mask=feature_mask,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    FEATURES_: tl.constexpr,
    NUM_CHUNKS_: tl.constexpr,
    FEATURE_BLOCK_: tl.constexpr,
    FINAL_CHUNK_BLOCK_: tl.constexpr,
):
    feature_block = tl.program_id(0)
    features = feature_block * FEATURE_BLOCK_ + tl.arange(0, FEATURE_BLOCK_)
    chunks = tl.arange(0, FINAL_CHUNK_BLOCK_)
    mask = (chunks[:, None] < NUM_CHUNKS_) & (features[None, :] < FEATURES_)
    values = tl.load(
        partial_ptr + chunks[:, None] * FEATURES_ + features[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(
        sum_ptr + features,
        total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
        mask=features < FEATURES_,
    )


# 4ad1eedb: timm SigLIP train two-source QK layout clone plus bf16-rounded sum.
@oracle_impl(hardware="B200", point="4ad1eedb")
def oracle_forward(inputs):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    layout = torch.empty_strided(
        (ROWS, FEATURES),
        (FEATURES, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (NUM_CHUNKS, FEATURES),
        (FEATURES, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((FEATURES,), (1,), device=arg0_1.device, dtype=torch.float32)

    _layout_partial_sum_kernel[(NUM_CHUNKS, triton.cdiv(FEATURES, FEATURE_BLOCK))](
        arg0_1,
        arg1_1,
        layout,
        partial,
        ROWS_=ROWS,
        FEATURES_=FEATURES,
        TOKENS_=TOKENS,
        INPUT_ROW_STRIDE_=INPUT_ROW_STRIDE,
        INPUT_HEAD_STRIDE_=INPUT_HEAD_STRIDE,
        INPUT_TOKEN_STRIDE_=INPUT_TOKEN_STRIDE,
        INPUT_DIM_STRIDE_=INPUT_DIM_STRIDE,
        SOURCE_COLS_=SOURCE_COLS,
        CHUNK_ROWS_=CHUNK_ROWS,
        ROW_BLOCK_=ROW_BLOCK,
        FEATURE_BLOCK_=FEATURE_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(FEATURES, FEATURE_BLOCK),)](
        partial,
        sum_out,
        FEATURES_=FEATURES,
        NUM_CHUNKS_=NUM_CHUNKS,
        FEATURE_BLOCK_=FEATURE_BLOCK,
        FINAL_CHUNK_BLOCK_=FINAL_CHUNK_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return layout, layout.permute(1, 0), sum_out
