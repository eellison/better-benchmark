"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT patch-padding scope in one Triton materialization kernel, returning both the observable all-zero bf16 `[512,4,256,40]` tensor and the separate bf16 slice_scatter result whose first 36 columns are the `[512,256,4,36] -> [512,4,256,36]` permutation of the input and whose last 4 columns remain zero; Inductor lowers the full/view/permute/slice_scatter graph through generic zero-fill and layout-scatter scheduling instead of sinking the final padded output indexing into one producer for both live outputs; the fix is SCHEDULER_FUSION: fuse the zero-fill side output with the padded permutation store while preserving the two non-aliasing contiguous return buffers."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 512
PATCHES = 4
TOKENS = 256
IN_PATCH_WIDTH = 36
OUT_PATCH_WIDTH = 40
HIDDEN = PATCHES * IN_PATCH_WIDTH
OUT_NUMEL = BATCH * PATCHES * TOKENS * OUT_PATCH_WIDTH


@triton.jit
def _zero_and_scatter_kernel(
    src_ptr,
    zero_out_ptr,
    scatter_out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    k = offsets % 40
    token = (offsets // 40) % 256
    patch = (offsets // (40 * 256)) % 4
    batch = offsets // (40 * 256 * 4)

    src_offsets = (batch * 256 + token) * 144 + patch * 36 + k
    has_source = k < 36

    zero = tl.full((BLOCK,), 0.0, tl.float32).to(tl.bfloat16)
    value = tl.load(src_ptr + src_offsets, mask=has_source, other=0.0).to(tl.bfloat16)
    scatter_value = tl.where(has_source, value, zero)

    tl.store(zero_out_ptr + offsets, zero)
    tl.store(scatter_out_ptr + offsets, scatter_value)


# 51a39cb3: MobileViT-S bf16 [131072,144] -> two [512,4,256,40] outputs.
@oracle_impl(hardware="B200", point="51a39cb3", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    src, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = (BATCH, PATCHES, TOKENS, OUT_PATCH_WIDTH)
    out_stride = (PATCHES * TOKENS * OUT_PATCH_WIDTH, TOKENS * OUT_PATCH_WIDTH, OUT_PATCH_WIDTH, 1)
    zero_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=src.device,
        dtype=torch.bfloat16,
    )
    scatter_out = torch.empty_strided(
        out_shape,
        out_stride,
        device=src.device,
        dtype=torch.bfloat16,
    )

    _zero_and_scatter_kernel[(triton.cdiv(OUT_NUMEL, BLOCK),)](
        src,
        zero_out,
        scatter_out,
        BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return zero_out, scatter_out
