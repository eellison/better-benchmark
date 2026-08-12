"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 DeepRecommender SELU pointwise scope `bf16(where(float32(x) > 0, float32(x) * 1.0507009873554805, expm1(float32(x) * 1.0) * 1.7580993408473766))` in one storage-linear Triton kernel while preserving the explicit fp32 compute and bf16 output rounding boundary, whereas Inductor lowers the isolated activation through generic fused pointwise codegen; Inductor cannot do this today because pointwise codegen has no dedicated B200-tuned exact-expm1 SELU activation template for this bf16 shape family; the fix is NEW_PATTERN: add a guarded exact-expm1 SELU pointwise specialization or equivalent autotuned launch choice when it beats generic pointwise codegen."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def _selu_bf16_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    positive = x * 1.0507009873554805
    negative = libdevice.expm1(x * 1.0) * 1.7580993408473766
    y = tl.where(x > 0.0, positive, negative)
    tl.store(output_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# 58f85f13: (T([256,197951], bf16))
@oracle_impl(hardware="B200", point="58f85f13")
# f1684e51: (T([256,512], bf16))
@oracle_impl(hardware="B200", point="f1684e51")
# 0f3e2fa1: (T([256,1024], bf16))
@oracle_impl(hardware="B200", point="0f3e2fa1")
def oracle_forward(inputs):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _selu_bf16_kernel[grid](
        x,
        out,
        n_elements=n_elements,
    )
    return out
