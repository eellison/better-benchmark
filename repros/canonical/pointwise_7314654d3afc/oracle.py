"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeiT bf16 classifier-logit averaging scope in one Triton pointwise kernel, including both contiguous `[128,1000]` bf16 input reads, the `aten.add.Tensor` bf16 rounding boundary, the `aten.div.Tensor(..., 2)` bf16 output rounding boundary, and the returned contiguous bf16 tensor, whereas Inductor already lowers this isolated graph to the same one-launch fused pointwise memory envelope; Inductor cannot materially improve this local repro through scheduler fusion because the full observable work is just two required input reads, one output store, and launch overhead with no avoidable materialization or layout work; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise case unless broader launch-overhead or tiny pointwise codegen changes move both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N_ELEMENTS = 128 * 1000


@triton.jit
def _add_div_bf16_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    total: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = (x + y).to(tl.bfloat16)
    out = (summed.to(tl.float32) / 2.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


# ced5d5ea: (T([128,1000], bf16), T([128,1000], bf16))
@oracle_impl(hardware="B200", point="ced5d5ea", BLOCK_N=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    x, y = inputs
    out = torch.empty_like(x)
    grid = (triton.cdiv(N_ELEMENTS, BLOCK_N),)
    _add_div_bf16_kernel[grid](
        x,
        y,
        out,
        total=N_ELEMENTS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out
