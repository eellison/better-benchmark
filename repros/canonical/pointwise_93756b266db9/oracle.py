"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete SigLIP bf16 bias-add plus reshape/permute scope by writing the contiguous backing storage for the rounded `[128,1,12,64]` add result in one Triton pointwise kernel and returning the eager-compatible `[128,12,1,64]` permute alias with stride `(768,64,768,1)`, whereas Inductor already lowers this tiny layout-only region to an equivalent fused pointwise/layout kernel; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, or algebraic elimination because the required work is one launch plus the mandatory activation read, broadcast-bias read, bf16 add/store, and metadata-only view return; the fix is BANDWIDTH_BOUND: record this as a layout pointwise floor unless broader launch-overhead or allocation work moves both paths together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bias_add_backing_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    total: tl.constexpr,
    row_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    cols = offsets % row_size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, (x + bias).to(tl.bfloat16), mask=mask)


@oracle_impl(
    hardware="B200",
    point="a6d7e457",
    BLOCK=256,
)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
):
    arg0_1, arg1_1, *_shape_params = inputs
    backing = torch.empty_strided(
        (128, 1, 12, 64),
        (768, 768, 64, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _bias_add_backing_kernel[(triton.cdiv(128 * 768, BLOCK),)](
        arg0_1,
        arg1_1,
        backing,
        total=128 * 768,
        row_size=768,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return backing.permute(0, 2, 1, 3)
