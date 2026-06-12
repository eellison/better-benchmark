"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete zero-rank `lift_fresh_copy` return for both bf16 and int64 scalar points by allocating one fresh scalar output and storing the input value with the original dtype, whereas Inductor lowers the same full scope through generic one-element pointwise codegen; Inductor cannot do this today because its scheduler/codegen path has no dedicated scalar fresh-copy materialization lowering that bypasses generic pointwise scheduling overhead while preserving non-aliasing output semantics; the fix is NEW_PATTERN: add a guarded zero-rank scalar fresh-copy lowering that emits the minimal scalar store path for dtype-preserving fresh tensor returns."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scalar_fresh_copy_kernel(input_ptr, output_ptr):
    value = tl.load(input_ptr)
    tl.store(output_ptr, value)


@oracle_impl(hardware="B200", point="2a837a19", num_warps=1)
@oracle_impl(hardware="B200", point="896c6bb5", num_warps=1)
def oracle_forward(inputs, *, num_warps):
    (arg0_1,) = inputs
    output = torch.empty_strided((), (), device=arg0_1.device, dtype=arg0_1.dtype)
    _scalar_fresh_copy_kernel[(1,)](arg0_1, output, num_warps=num_warps)
    return output
