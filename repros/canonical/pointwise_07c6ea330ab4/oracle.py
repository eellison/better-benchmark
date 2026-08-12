"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete bf16 Longformer bias add, head split/permutation, symmetric constant pad, overlapping `as_strided` clone, and final contiguous `[384,768,64]` chunk layout into one Triton materialization kernel; Inductor currently schedules the pointwise add, padding, and overlapping clone/layout materialization through separate generic regions; Inductor cannot do this today because clone/as_strided materialization with padded affine indexing is treated as a scheduler fusion barrier even though the full scope is one affine output-space map; the fix is SCHEDULER_FUSION: teach layout clone codegen to sink bf16 pointwise producers and constant padding into affine overlapping-window materialization kernels that write the final contiguous chunk buffer directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 1024
BATCH = 8
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
HEAD_BATCH = BATCH * HEADS
WINDOWS = 4
WINDOW_SIZE = 768
WINDOW_STEP = 256
PAD_BEFORE = 256
OUTPUT_SHAPE = (HEAD_BATCH * WINDOWS, WINDOW_SIZE, HEAD_DIM)
OUTPUT_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)


@triton.jit
def _longformer_padded_chunk_layout_kernel(
    input_ptr,
    bias_ptr,
    out_ptr,
    TOTAL_CHUNKS: tl.constexpr,
    WINDOW_SIZE_C: tl.constexpr,
    WINDOW_STEP_C: tl.constexpr,
    PAD_BEFORE_C: tl.constexpr,
    SEQ_C: tl.constexpr,
    BATCH_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    HEADS_C: tl.constexpr,
    HEAD_DIM_C: tl.constexpr,
    WINDOWS_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(0)
    pos = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    dim = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    head_batch = chunk // WINDOWS_C
    window = chunk - head_batch * WINDOWS_C
    batch = head_batch // HEADS_C
    head = head_batch - batch * HEADS_C

    source_seq = pos + window * WINDOW_STEP_C - PAD_BEFORE_C
    source_feature = head * HEAD_DIM_C + dim
    load_offsets = (source_seq[:, None] * BATCH_C + batch) * HIDDEN_C + source_feature[None, :]
    store_offsets = chunk * WINDOW_SIZE_C * HEAD_DIM_C + pos[:, None] * HEAD_DIM_C + dim[None, :]

    output_mask = (
        (chunk < TOTAL_CHUNKS)
        & (pos[:, None] < WINDOW_SIZE_C)
        & (dim[None, :] < HEAD_DIM_C)
    )
    valid = output_mask & (source_seq[:, None] >= 0) & (source_seq[:, None] < SEQ_C)

    values = tl.load(input_ptr + load_offsets, mask=valid, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + source_feature, mask=dim < HEAD_DIM_C, other=0.0).to(tl.float32)
    out = tl.where(valid, values + bias[None, :], -1.0)
    tl.store(out_ptr + store_offsets, out, mask=output_mask)


@oracle_impl(hardware="B200", point="5fa3702b", BLOCK_P=32, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_D: int, num_warps: int):
    arg0_1, arg1_1, *_shape_params = inputs
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = (
        OUTPUT_SHAPE[0],
        triton.cdiv(WINDOW_SIZE, BLOCK_P),
        triton.cdiv(HEAD_DIM, BLOCK_D),
    )
    _longformer_padded_chunk_layout_kernel[grid](
        arg0_1,
        arg1_1,
        out,
        TOTAL_CHUNKS=OUTPUT_SHAPE[0],
        WINDOW_SIZE_C=WINDOW_SIZE,
        WINDOW_STEP_C=WINDOW_STEP,
        PAD_BEFORE_C=PAD_BEFORE,
        SEQ_C=SEQ,
        BATCH_C=BATCH,
        HIDDEN_C=HIDDEN,
        HEADS_C=HEADS,
        HEAD_DIM_C=HEAD_DIM,
        WINDOWS_C=WINDOWS,
        BLOCK_P=BLOCK_P,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
