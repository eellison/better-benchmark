"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete AllenAI Longformer bf16 `full(-0.0)` plus two singleton-select scope by materializing the required fresh `bfloat16[8,1024]` result with one Triton negative-zero bit-fill kernel, whereas Inductor lowers the captured constant full/select/select graph through generic constant materialization and view planning; Inductor cannot do this today because it has no guarded full-plus-singleton-select materialization template that writes the final observed layout and preserves bf16 negative-zero bits directly; the fix is SCHEDULER_FUSION: fuse constant full and metadata-only selects into a direct final-output fill for this layout-indexing pattern."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUT_SHAPE = (8, 1024)
OUT_STRIDE = (1024, 1)
OUT_NUMEL = 8 * 1024


@triton.jit
def _fill_negative_zero_bf16_kernel(
    out_bits_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total
    values = tl.full((BLOCK,), 0x8000, tl.uint16)
    tl.store(out_bits_ptr + offsets, values, mask=active)


# d7517139: Longformer bf16 full(-0.0)[8,1,1,1024] -> select -> [8,1024].
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    _shape_param_0, = inputs
    del _shape_param_0
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    _fill_negative_zero_bf16_kernel[(triton.cdiv(OUT_NUMEL, BLOCK),)](
        out.view(torch.uint16),
        total=OUT_NUMEL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
