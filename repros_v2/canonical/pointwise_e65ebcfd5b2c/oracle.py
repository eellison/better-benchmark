"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 DCGAN two-input leaky-gate scope in one storage-linear Triton pointwise kernel, including promotion of the value and gate tensors to fp32, the `gate > 0` predicate, the fp32 `value * 0.2` negative branch, `where` selection, and final bf16 contiguous output store, whereas Inductor already lowers this isolated activation as a comparable fused pointwise kernel; Inductor cannot materially reduce this local repro with scheduler fusion, algebraic elimination, split-K, scatter-reduce, or a new stencil pattern because the full scope is dominated by mandatory dense reads and output write with no adjacent producer or consumer to eliminate; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise case unless broader pointwise launch or memory-throughput improvements move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _leaky_gate_bf16_kernel(
    value_ptr,
    gate_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(value_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    negative = value * 0.2
    out = tl.where(gate > 0.0, value, negative)
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="78d93924", BLOCK=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    value, gate = inputs
    out = torch.empty_strided(
        tuple(value.shape),
        tuple(value.stride()),
        device=value.device,
        dtype=torch.bfloat16,
    )
    n_elements = value.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _leaky_gate_bf16_kernel[grid](
        value,
        gate,
        out,
        n_elements=n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
