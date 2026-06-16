"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete f32-to-bf16 cast plus metadata transpose scope by writing the fresh contiguous bf16 base tensor in one storage-linear Triton pointwise kernel and returning its `permute(1, 0)` view, whereas Inductor already lowers this isolated convert-plus-view graph to the same mandatory f32 read, bf16 write, and alias-view output envelope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because there are no surrounding producers or consumers to fuse and the transpose is only output metadata; the fix is BANDWIDTH_BOUND: record this as a pointwise cast bandwidth/launch floor unless broader pointwise codegen or launch-overhead work moves both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_to_bf16_base_kernel(
    input_ptr,
    base_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(base_ptr + offsets, values.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="af408fe3", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="fddbd2f3", BLOCK=16, num_warps=1)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (x,) = inputs
    base = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    _f32_to_bf16_base_kernel[(triton.cdiv(n_elements, BLOCK),)](
        x,
        base,
        n_elements=n_elements,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return base.permute(1, 0)
