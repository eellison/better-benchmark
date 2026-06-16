"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Lennard-Jones bf16 producer in one Triton tile, preserving the captured f32-to-bf16 round trip on the `[1,16]` coefficient row, the bf16 round trip after the first multiply, the final bf16 materialization, the returned transpose alias, and the final f32-sum-to-bf16-to-f32 column-sum boundary. Inductor schedules the small broadcast producer, required bf16 casts, returned transpose view, and sibling column reduction through generic materialization/reduction code; it cannot do this today because the scheduler does not form one alias-aware full-scope producer-store plus reduction plan for a rounded value that is both returned and reduced. The fix is SCHEDULER_FUSION: keep the rounded producer virtual, emit the visible base storage once, return metadata-only aliases, and finalize compatible column reductions from the same traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 128
N = 16


@triton.jit
def _lennard_jones_kernel(
    row_ptr,
    coeff_ptr,
    gate_ptr,
    out_ptr,
    sum_ptr,
    N_: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    offsets = rows * N_ + col

    row = tl.load(row_ptr + rows).to(tl.float32)
    coeff = tl.load(coeff_ptr + col).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    first = (row * coeff).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    gate = tl.load(gate_ptr + offsets).to(tl.float32)
    gate_sq = gate * gate
    complement = 1.0 - gate_sq
    out = (first * complement).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, out)
    total = tl.sum(out.to(tl.float32), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + col, rounded)


# torchbench_lennard_jones train, M=128 N=16.
@oracle_impl(hardware="B200", point="395f4c9c", BLOCK_M=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    row, coeff, gate, sum_shape_arg = inputs

    out = torch.empty_strided((M, N), (N, 1), device=row.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape_arg),
        (1,),
        device=row.device,
        dtype=torch.float32,
    )

    _lennard_jones_kernel[(N,)](
        row,
        coeff,
        gate,
        out,
        sum_out,
        N_=N,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=1,
    )
    return out, torch.as_strided(out, (N, M), (1, N)), sum_out
