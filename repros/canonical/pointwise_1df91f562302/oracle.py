"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 NFNet exact-erf GELU scale scope in one storage-linear Triton kernel, preserving the fp32 `0.5*x*(erf(0.7071067811865476*x)+1)` computation, the explicit bf16 rounding boundary, the final `1.7015043497085571` bf16 output multiply, and the captured channels-last dense output layout, whereas Inductor lowers the same isolated activation through its generic fused pointwise path; Inductor cannot do this today because pointwise codegen has no dedicated B200-tuned exact-GELU-scale bf16 template for this NFNet shape family; the fix is NEW_PATTERN: add a guarded exact-GELU-scale pointwise specialization or equivalent autotuned launch choice when it beats generic pointwise codegen."""

import torch
import triton
import triton.language as tl

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
def _nfnet_gelu_scale_bf16_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(input_ptr + offsets).to(tl.float32)
    gelu = (x * 0.5) * (tl.math.erf(x * 0.7071067811865476) + 1.0)
    gelu_bf16 = gelu.to(tl.bfloat16)
    scaled = gelu_bf16.to(tl.float32) * 1.7015043497085571
    tl.store(output_ptr + offsets, scaled.to(tl.bfloat16))


@oracle_impl(hardware="B200", point="4c47a3fc")
@oracle_impl(hardware="B200", point="192c8e99")
@oracle_impl(hardware="B200", point="b00342cf")
@oracle_impl(hardware="B200", point="f77e434f")
@oracle_impl(hardware="B200", point="209c35c7")
@oracle_impl(hardware="B200", point="0630f81f")
@oracle_impl(hardware="B200", point="926b386f")
@oracle_impl(hardware="B200", point="95428590")
@oracle_impl(hardware="B200", point="b358c4cf")
@oracle_impl(hardware="B200", point="1acf97f6")
@oracle_impl(hardware="B200", point="6fef8a13")
@oracle_impl(hardware="B200", point="20ba0bab")
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
    _nfnet_gelu_scale_bf16_kernel[grid](
        x,
        out,
        n_elements,
    )
    return out
