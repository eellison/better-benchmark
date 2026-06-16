"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete AllenAI Longformer `full(-0.0)` plus two singleton-select scope by materializing the required fresh `float32[8,1024]` result with one Triton negative-zero fill kernel, whereas Inductor lowers the captured full/select/select graph through generic constant materialization and view planning; Inductor cannot materially improve this isolated repro today because the only observable work is a fresh tiny output allocation, one launch, and the required negative-zero stores; the fix is BANDWIDTH_BOUND: record this as a launch/materialization floor unless broader constant-output allocation or launch overhead changes move both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUT_SHAPE = (8, 1024)
OUT_STRIDE = (1024, 1)
OUT_NUMEL = 8 * 1024


@triton.jit
def _fill_negative_zero_kernel(
    out_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total
    values = tl.full((BLOCK,), -2147483648, tl.int32).to(tl.float32, bitcast=True)
    tl.store(out_ptr + offsets, values, mask=active)


# d7517139: Longformer f32 full(-0.0)[8,1,1,1024] -> select -> [8,1024].
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    _shape_param_0, = inputs
    del _shape_param_0
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    _fill_negative_zero_kernel[(triton.cdiv(OUT_NUMEL, BLOCK),)](
        out,
        total=OUT_NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
