"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete no-input scalar_tensor scope by allocating the fresh zero-dimensional cuda bf16 result and storing the exact 0.0 value with one minimal Triton program, whereas Inductor currently lowers aten.scalar_tensor through its generic scalar tensor creation path at launch floor; Inductor cannot do this today because its scheduler/codegen pattern library has no dedicated zero-dimensional CUDA scalar constant creation lowering that bypasses generic pointwise machinery while preserving the fresh output tensor; the fix is NEW_PATTERN: add an Inductor scalar constant-tensor lowering for static aten.scalar_tensor that emits a minimal scalar store or backend scalar-fill primitive directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scalar_bf16_zero_kernel(out_ptr):
    tl.store(out_ptr, tl.full((), 0.0, tl.float32))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    out = torch.empty((), device=torch.device("cuda", 0), dtype=torch.bfloat16)
    _scalar_bf16_zero_kernel[(1,)](out, num_warps=1)
    return out
