"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Lennard-Jones bf16 `[128,1]` column sum in one Triton reduction, including fp32 accumulation for `sum(dim=0, keepdim=True, dtype=float32)`, the metadata-only `[1,1] -> [1]` view, and the explicit f32-to-bf16-to-f32 output round trip, whereas Inductor lowers the tiny one-column reduction and cast epilogue through generic reduction scheduling; Inductor cannot do this today because its scheduler/codegen has no dedicated fixed-extent one-column bf16 reduction template that strips generic reduction scaffolding while preserving the captured output rounding boundary; the fix is NEW_PATTERN: add a guarded small-column sum lowering that emits the compact reduction and bf16-rounding epilogue directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _sum_128x1_roundtrip_kernel(x_ptr, out_ptr, BLOCK_M: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    values = tl.load(x_ptr + rows).to(tl.float32)
    total = tl.sum(values, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_ptr, rounded)


@oracle_impl(hardware="B200", point="10254c9c", BLOCK_M=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    (arg0_1,) = inputs
    out = torch.empty_strided((1,), (1,), device=arg0_1.device, dtype=torch.float32)
    _sum_128x1_roundtrip_kernel[(1,)](
        arg0_1,
        out,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
