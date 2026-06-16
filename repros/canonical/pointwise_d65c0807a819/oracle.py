"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bf16 bias cast/add, head/batch layout split, overlapping three-window as_strided stencil materialization, contiguous clone, and returned permuted view by writing the `[288, 64, 512]` clone storage directly and returning its metadata-only `[288, 512, 64]` view, whereas Inductor lowers the decomposed view/permute/as_strided/clone chain through generic pointwise and layout materialization code; Inductor cannot do this today because the scheduler treats the overlapping as_strided clone as a fusion barrier for the upstream bias-add producer and does not canonicalize this fixed Longformer sliding-window layout as one direct indexed store; the fix is SCHEDULER_FUSION: teach layout clone codegen to sink affine pointwise producers through overlapping as_strided window indexing and preserve returned aliasing views."""

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
WINDOWS = 3
WINDOW_SIZE = 512
WINDOW_STEP = 256
OUT_SHAPE = (HEAD_BATCH * WINDOWS, HEAD_DIM, WINDOW_SIZE)
OUT_STRIDE = (HEAD_DIM * WINDOW_SIZE, WINDOW_SIZE, 1)


@triton.jit
def _longformer_bf16_layout_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dim_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    chunk = tl.program_id(2)

    head_batch = chunk // 3
    window = chunk - head_batch * 3
    head = head_batch % 12

    source_token = window * 256 + token_offsets
    source_feature = head * 64 + dim_offsets
    source_offsets = (
        source_token[:, None] * 6144
        + head_batch * 64
        + dim_offsets[None, :]
    )
    load_mask = (token_offsets[:, None] < 512) & (dim_offsets[None, :] < 64)

    values = tl.load(x_ptr + source_offsets, mask=load_mask, other=0.0).to(tl.float32)
    bias = tl.load(
        bias_ptr + source_feature,
        mask=dim_offsets < 64,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.bfloat16).to(tl.float32)
    out = values + bias[None, :]

    store_offsets = (
        chunk * 32768
        + dim_offsets[:, None] * 512
        + token_offsets[None, :]
    )
    store_mask = (dim_offsets[:, None] < 64) & (token_offsets[None, :] < 512)
    tl.store(out_ptr + store_offsets, tl.trans(out), mask=store_mask)


@oracle_impl(hardware="B200", point="53c69788", BLOCK_T=32, BLOCK_D=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_T: int, BLOCK_D: int, num_warps: int):
    bias, x, *_shape_params = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (
        triton.cdiv(WINDOW_SIZE, BLOCK_T),
        triton.cdiv(HEAD_DIM, BLOCK_D),
        HEAD_BATCH * WINDOWS,
    )
    _longformer_bf16_layout_kernel[grid](
        x,
        bias,
        out,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.permute(0, 2, 1)
