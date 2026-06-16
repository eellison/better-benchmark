"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DeBERTa attention-head clone/div/permute scope by writing the divided contiguous `[B*H, S, D]` backing storage once and returning both the `[B*H, D, S]` view and its `[B*H, S, D]` permutation, whereas Inductor lowers the captured view/permute/clone/div/permute chain through generic layout and pointwise scheduling; Inductor cannot do this today because the scheduler does not have a single alias-aware layout-divide template that materializes the required backing storage while exposing both returned views; the fix is SCHEDULER_FUSION: add a guarded head-layout materialization template that fuses the scalar divide into the clone backing store and preserves the visible view aliases."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _deberta_head_div_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    head_dim = offsets % 64
    seq = (offsets // 64) % 512
    batch_head = offsets // 32768

    x_offsets = (
        head_dim
        + 64 * (batch_head % 24)
        + 1536 * seq
        + 786432 * (batch_head // 24)
    )

    x = tl.load(x_ptr + x_offsets).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    y = x / scale
    tl.store(out_ptr + offsets, y)


# 6bfc0be7: (T([4096,1536], bf16), T([], bf16), S([8,512,1536]), ...)
@oracle_impl(hardware="B200", point="6bfc0be7", BLOCK=2048, num_warps=1)
def oracle_forward(inputs, *, BLOCK, num_warps):
    x, scale, _shape0, _shape1, _shape2 = inputs
    base = torch.empty_strided(
        (192, 512, 64),
        (32768, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(6291456, BLOCK),)
    _deberta_head_div_kernel[grid](
        x,
        scale,
        base,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return base.permute(0, 2, 1), base
