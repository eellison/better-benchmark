"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete no-input `aten.full.default([], 4096, dtype=int64)` scope by allocating the fresh zero-dimensional CUDA int64 result and storing the exact scalar value with one minimal Triton program, whereas Inductor currently lowers the standalone scalar full through its generic constant pointwise codegen path; Inductor cannot do this today because its scheduler/codegen pattern library has no dedicated zero-dimensional CUDA `aten.full` constant creation lowering that bypasses generic pointwise machinery while preserving the fresh output tensor; the fix is NEW_PATTERN: add a guarded scalar constant-full lowering that emits a minimal scalar store or backend scalar-fill primitive directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scalar_i64_full_kernel(out_ptr):
    tl.store(out_ptr, tl.full((), 4096, tl.int64))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    out = torch.empty((), device=torch.device("cuda", 0), dtype=torch.int64)
    _scalar_i64_full_kernel[(1,)](out, num_warps=1)
    return out
