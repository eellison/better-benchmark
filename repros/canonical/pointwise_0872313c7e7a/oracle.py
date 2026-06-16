"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 DCGAN leaky-ReLU scope in one storage-linear Triton pointwise kernel, including bf16 input promotion to fp32, the `x > 0` predicate, fp32 `x * 0.2` negative branch, `where` selection, and final bf16 contiguous output store, whereas Inductor already lowers this isolated activation as a comparable fused pointwise kernel; Inductor cannot materially reduce this today because the full scope is dominated by the mandatory dense input read and output write with no producer, reduction, scatter, or algebraic work to eliminate; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise case unless generic pointwise launch or memory-throughput improvements move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _leaky_relu_bf16_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    y = tl.where(x > 0.0, x, x * 0.2)
    tl.store(out_ptr + offsets, y)


# eb973076: bf16[256,64,32,32], contiguous
@oracle_impl(hardware="B200", point="eb973076", BLOCK=1024, num_warps=4)
# 356e4166: bf16[32,64,32,32], contiguous
@oracle_impl(hardware="B200", point="356e4166", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _leaky_relu_bf16_kernel[grid](
        x,
        out,
        n_elements=n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
