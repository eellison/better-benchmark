"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full metadata-view plus bf16-to-fp32 add scope as one storage-linear Triton pointwise kernel over the contiguous inputs and fresh contiguous rank-3 output, whereas Inductor lowers the public 3D views, two dtype conversions, and add through a generic pointwise path with extra view/shape handling; Inductor cannot do this today because pointwise codegen does not consistently canonicalize metadata-only view pairs with dtype promotion into a mask-free storage-linear add when the observable output rank changes; the fix is NEW_PATTERN: add a view-linearization pointwise canonicalization that emits direct contiguous indexing for metadata-only views before scheduling/codegen while preserving bf16 input promotion and fp32 output storage."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bf16_add_views_to_fp32_kernel(
    left_ptr,
    right_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        left = tl.load(left_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(right_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, left + right, mask=mask)
    else:
        left = tl.load(left_ptr + offsets).to(tl.float32)
        right = tl.load(right_ptr + offsets).to(tl.float32)
        tl.store(out_ptr + offsets, left + right)


def _contiguous_3d_stride(shape):
    return (int(shape[1]) * int(shape[2]), int(shape[2]), 1)


@oracle_impl(hardware="B200", point="a3b18231", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="e47867ab", BLOCK_SIZE=2048, USE_MASK=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, USE_MASK: bool, num_warps: int):
    left, right, out_shape, _out_shape_1 = inputs
    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=left.device,
        dtype=torch.float32,
    )
    n_elements = left.numel()
    _bf16_add_views_to_fp32_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        left,
        right,
        out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
