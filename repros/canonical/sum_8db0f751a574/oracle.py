"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 attention-probability backward row update in one Triton row-reduction kernel, preserving the bool-to-bf16 dropout-scale multiply, the bf16 product rounding before fp32 accumulation, the exact fma.rn.f32 epilogue, and the final bf16 view-shaped store, whereas Inductor currently lowers the decomposed view/cast/mul/sum/neg/fma/cast chain through generic reduction scheduling with explicit bf16 rounding boundaries around the producer; Inductor cannot do this today because its scheduler/codegen does not keep this bf16-rounded producer, row reduction, and dependent fma epilogue in one B200-tuned row program across the captured attention shape family; the fix is SCHEDULER_FUSION: add a row-reduction fusion template that preserves explicit bf16 casts while sinking the producer and fma consumer into one schedule."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_sum_fma_bf16_kernel(
    grad_ptr,
    keep_ptr,
    prob_ptr,
    out_ptr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * K + cols

    grad = tl.load(grad_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    keep = tl.load(keep_ptr + offsets, eviction_policy="evict_first").to(tl.int1)
    prob = tl.load(prob_ptr + offsets, eviction_policy="evict_first").to(tl.float32)

    keep_scale = (keep.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    scaled_grad = (grad * keep_scale).to(tl.bfloat16).to(tl.float32)
    product = scaled_grad * prob
    row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)
    out = _fma_rn_f32(-prob, row_sum, product)

    tl.store(out_ptr + offsets, out.to(tl.bfloat16))


def _launch_row_sum_fma(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    grad, keep, prob, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    k = prob.shape[-1]
    rows = prob.numel() // k
    grid = (triton.cdiv(rows, BLOCK_M),)
    _row_sum_fma_bf16_kernel[grid](
        grad,
        keep,
        prob,
        out,
        K=k,
        SCALE=DROPOUT_SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


# (T([1024, 128, 128], bf16), T([64, 16, 128, 128], b8), T([64, 16, 128, 128], bf16), S([64, 16, 128, 128]), S([1024, 128, 128]))
@oracle_impl(hardware="B200", point="2e48b06a", BLOCK_M=8, BLOCK_N=128, num_warps=8)
# (T([1024, 128, 128], bf16), T([256, 4, 128, 128], b8), T([256, 4, 128, 128], bf16), S([256, 4, 128, 128]), S([1024, 128, 128]))
@oracle_impl(hardware="B200", point="87ed7153", BLOCK_M=8, BLOCK_N=128, num_warps=4)
# (T([3072, 128, 128], bf16), T([256, 12, 128, 128], b8), T([256, 12, 128, 128], bf16), S([256, 12, 128, 128]), S([3072, 128, 128]))
@oracle_impl(hardware="B200", point="6ffb5b71", BLOCK_M=8, BLOCK_N=128, num_warps=4)
# (T([256, 512, 512], bf16), T([64, 4, 512, 512], b8), T([64, 4, 512, 512], bf16), S([64, 4, 512, 512]), S([256, 512, 512]))
@oracle_impl(hardware="B200", point="c99a78fd", BLOCK_M=2, BLOCK_N=512, num_warps=4)
# (T([384, 512, 512], bf16), T([32, 12, 512, 512], b8), T([32, 12, 512, 512], bf16), S([32, 12, 512, 512]), S([384, 512, 512]))
@oracle_impl(hardware="B200", point="d02283d5", BLOCK_M=2, BLOCK_N=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    return _launch_row_sum_fma(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
