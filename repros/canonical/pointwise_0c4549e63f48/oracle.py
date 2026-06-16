"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete zero-input ConvBERT inference layout-indexing scope by materializing the required CUDA `int64[1]` iota buffer with value zero and returning both that tensor and its `unsqueeze(0)` `[1,1]` view; Inductor already lowers this tiny `prims.iota` plus metadata-only view to launch-scale work, and there is no producer/consumer chain or algebraic consumer to fuse away, so the fix class is BANDWIDTH_BOUND: the oracle records the launch-floor full-scope cost while preserving the returned aliasing/layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _zero_i64_kernel(out_ptr):
    tl.store(out_ptr, tl.full((), 0, tl.int64))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs

    iota = torch.empty_strided((1,), (1,), device=torch.device("cuda", 0), dtype=torch.int64)
    _zero_i64_kernel[(1,)](iota, num_warps=1, num_stages=1)
    return iota, iota.unsqueeze(0)
