"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete zero-input ConvBERT index-buffer scope by materializing the length-1 CUDA int64 iota with one Triton scalar store and returning the metadata-only `unsqueeze(-1)` alias from that buffer, whereas Inductor lowers the captured `prims.iota` plus view through its generic pointwise/output path; Inductor cannot do this today because it has no guarded zero-input scalar-iota materialization template that also preserves the returned alias-view contract; the fix is NEW_PATTERN: add a scalar iota/view lowering for launch-floor index-buffer graphs."""

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

    iota = torch.empty_strided(
        (1,),
        (1,),
        device=torch.device("cuda", 0),
        dtype=torch.int64,
    )
    _zero_i64_kernel[(1,)](iota, num_warps=1, num_stages=1)
    return iota, iota.unsqueeze(-1)
