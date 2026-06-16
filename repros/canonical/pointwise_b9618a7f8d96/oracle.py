"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete scalar float32 `lift_fresh_copy`, identity multiply, and `sqrt` return scope from one loaded scalar and three fresh scalar stores, whereas Inductor lowers the captured graph through a generic one-element pointwise Triton kernel for the materialized scalar outputs; Inductor cannot do this today because its pointwise scheduler does not canonicalize scalar fresh-copy identity arithmetic into a dedicated scalar materialization path while preserving the visible multi-output scope; the fix is ALGEBRAIC_ELIMINATION: fold `mul(x, 1)` in scalar fresh-output graphs and lower the remaining scalar copy/sqrt stores through a compact scalar template instead of generic pointwise scheduling."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scalar_copy_mul_sqrt_kernel(input_ptr, copy_ptr, mul_ptr, sqrt_ptr):
    value = tl.load(input_ptr).to(tl.float32)
    mul = value * 1.0
    tl.store(copy_ptr, value)
    tl.store(mul_ptr, mul)
    tl.store(sqrt_ptr, tl.sqrt_rn(mul))


# 406cae07: (T([], f32))
@oracle_impl(hardware="B200", point="406cae07", num_warps=1)
def oracle_forward(inputs, *, num_warps: int):
    (arg0_1,) = inputs
    copy = torch.empty_like(arg0_1)
    mul = torch.empty_like(arg0_1)
    sqrt = torch.empty_like(arg0_1)
    _scalar_copy_mul_sqrt_kernel[(1,)](
        arg0_1,
        copy,
        mul,
        sqrt,
        num_warps=num_warps,
        num_stages=1,
    )
    return copy, mul, sqrt
