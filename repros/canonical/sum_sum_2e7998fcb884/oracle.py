"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ALBERT bf16 dual-row-sum backward fragment in one row-resident Triton kernel, sharing the three casted bf16 matrix adds, the two hidden-dimension fp32 reductions, the dependent fp32 epilogue, and the required fp32/bf16 side-output materializations including the final permuted view, whereas Inductor schedules the reduction-dependent expression and returned side outputs through generic fusion boundaries that cannot keep every row scalar and dtype-cast output in one full-scope plan; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output row-reduction template that preserves intermediate returned tensors, bf16 rounding boundaries, and metadata-alias view returns together; the fix is SCHEDULER_FUSION: add a guarded row-resident multi-output reduction schedule for layer-norm-backward-style fragments that shares row reductions and sinks all returned epilogues into the same generated kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 8 * 512
HIDDEN = 4096


@triton.jit
def _row_resident_dual_sum_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    add_out_ptr,
    mul_out_ptr,
    bf16_out_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN_ + cols

    base = tl.load(arg1_ptr + offsets).to(tl.float32)
    add0 = base + tl.load(arg0_ptr + offsets).to(tl.float32)
    add1 = add0 + tl.load(arg2_ptr + offsets).to(tl.float32)
    add2 = add1 + tl.load(arg3_ptr + offsets).to(tl.float32)
    tl.store(add_out_ptr + offsets, add2)

    scale = tl.load(arg4_ptr + cols).to(tl.float32)
    mul = add2 * scale
    rhs = tl.load(arg5_ptr + offsets).to(tl.float32)
    sum0 = tl.sum(mul, axis=0)
    sum1 = tl.sum(mul * rhs, axis=0)

    row_scale = tl.load(arg6_ptr + row).to(tl.float32)
    out = row_scale * ((mul * HIDDEN_) - sum0 - (rhs * sum1))
    tl.store(mul_out_ptr + offsets, out)
    tl.store(bf16_out_ptr + offsets, out.to(tl.bfloat16))


@oracle_impl(hardware="B200", point="b3c0b271", BLOCK_H=HIDDEN, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        shape_param_3,
    ) = inputs

    add_out = torch.empty_like(arg1_1)
    mul_out = torch.empty_like(arg5_1)
    bf16_out = torch.empty_strided(
        tuple(shape_param_3),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _row_resident_dual_sum_kernel[(ROWS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        add_out,
        mul_out,
        bf16_out,
        HIDDEN_=HIDDEN,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )

    return add_out, mul_out, bf16_out, bf16_out.permute(1, 0)
