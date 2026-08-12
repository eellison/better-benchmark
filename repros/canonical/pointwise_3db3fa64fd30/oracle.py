"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete shape-param `aten.full.default([4, 2048], 1, dtype=float32)` scope by allocating the fresh contiguous CUDA f32 output and writing `1.0` with one minimal Triton fill kernel, whereas Inductor lowers the standalone full through its generic no-load pointwise constant-store scheduler; Inductor cannot do this today because codegen has no dedicated static constant-full lowering that bypasses generic elementwise indexing for shape-only full tensors; the fix is NEW_PATTERN: add a guarded constant-full template for CUDA `aten.full` that writes the required output layout directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _fill_one_f32_kernel(
    out_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    values = tl.full((BLOCK,), 1.0, tl.float32)
    tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (_shape_param_0,) = inputs
    shape = tuple(int(dim) for dim in _shape_param_0)
    out = torch.empty_strided(
        shape,
        (shape[1], 1),
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    total = out.numel()
    _fill_one_f32_kernel[(triton.cdiv(total, BLOCK),)](
        out,
        total=total,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
