"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete zero-input Whisper singleton iota plus add-zero scope by materializing two separate CUDA `int64[1]` buffers with value zero in one Triton launch, preserving the fresh `aten.add(iota, 0)` output rather than aliasing the iota tensor, whereas Inductor lowers the scalar iota and algebraically neutral add through generic tiny pointwise/layout scheduling; Inductor cannot do this today because its simplifier does not fold singleton generated-index add-zero into direct constant materialization with the same two-output storage contract; the fix is ALGEBRAIC_ELIMINATION: eliminate the add-zero while emitting both required output buffers directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _zero_two_i64_kernel(iota_ptr, add_ptr):
    zero = tl.full((), 0, tl.int64)
    tl.store(iota_ptr, zero)
    tl.store(add_ptr, zero)


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs

    device = torch.device("cuda", 0)
    iota = torch.empty_strided((1,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((1,), (1,), device=device, dtype=torch.int64)
    _zero_two_i64_kernel[(1,)](iota, add, num_warps=1, num_stages=1)
    return iota, add
