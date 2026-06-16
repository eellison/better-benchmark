"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete attention-mask bias materialization in one flat Triton pass, loading the captured bool mask once and writing the required bf16 `0.0` or `-inf` sentinel output directly, whereas Inductor already lowers the two scalar fills plus `aten.where` as equivalent generic pointwise materialization; Inductor cannot materially improve this today because the region has no producer, consumer, algebra, or layout transform to eliminate and is dominated by one launch plus the required mask read and bf16 store; the fix is BANDWIDTH_BOUND: record this as an at-floor sentinel materialization unless broader launch/allocation or pointwise codegen overhead changes the baseline."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _mask_to_bias_kernel(mask_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    mask = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    out = tl.where(mask, 0.0, -float("inf"))
    tl.store(out_ptr + offsets, out, mask=valid)


@oracle_impl(hardware="B200", point="605f6819", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="6fdaf67b", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="08fb6d2d", BLOCK=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="9ba6de11", BLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    (mask,) = inputs
    out = torch.empty_strided(
        tuple(mask.shape),
        tuple(mask.stride()),
        device=mask.device,
        dtype=torch.bfloat16,
    )
    _mask_to_bias_kernel[(triton.cdiv(mask.numel(), BLOCK),)](
        mask,
        out,
        N=mask.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
