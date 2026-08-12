"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete VGG bf16 masked-dropout producer once, returns the observable bf16 scalar zero, writes the returned contiguous `[64,4096]` bf16 tensor, returns the required metadata-only transpose alias, and accumulates the sibling dim-0 f32 column sum from the same bf16-rounded producer values before applying the captured final bf16-to-f32 round-trip. Inductor lowers the shared `where(mask, full, bf16_input * bf16_bool * 2)`, transpose view, and column reduction through generic multi-output reduction scheduling with avoidable producer replay/materialization overhead; it cannot do this today because the scheduler does not form an alias-aware full-scope producer-store plus reduction plan for a value that is both returned and reduced while preserving explicit bf16 rounding boundaries and scalar output scope. The fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to keep the masked bf16 producer virtual, emit the visible base storage once, return view aliases, and finalize compatible column reductions from the same traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 64
N = 4096


@triton.jit
def _masked_store_sum_kernel(
    scale_mask_ptr,
    x_ptr,
    fill_mask_ptr,
    fill_out_ptr,
    out_ptr,
    sum_ptr,
    M_: tl.constexpr,
    N_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    active = (rows[:, None] < M_) & (cols[None, :] < N_)
    offsets = rows[:, None] * N_ + cols[None, :]

    tl.store(fill_out_ptr, tl.full((), 0.0, tl.float32).to(tl.bfloat16), mask=pid == 0)

    scale_mask = tl.load(scale_mask_ptr + offsets, mask=active, other=0)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill_mask = tl.load(fill_mask_ptr + offsets, mask=active, other=0)

    scaled = (x * tl.where(scale_mask != 0, 2.0, 0.0)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    zero = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32).to(tl.bfloat16)
    values = tl.where(fill_mask != 0, zero, scaled)

    tl.store(out_ptr + offsets, values, mask=active)
    total = tl.sum(tl.where(active, values.to(tl.float32), 0.0), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded, mask=cols < N_)


# torchbench_vgg16 train, M=64 N=4096.
@oracle_impl(hardware="B200", point="c8695fa1", BLOCK_M=64, BLOCK_N=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    scale_mask, x, fill_mask, sum_shape_arg = inputs

    fill_out = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)
    out = torch.empty_strided((M, N), (N, 1), device=x.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in sum_shape_arg),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )

    _masked_store_sum_kernel[(triton.cdiv(N, BLOCK_N),)](
        scale_mask,
        x,
        fill_mask,
        fill_out,
        out,
        sum_out,
        M_=M,
        N_=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return fill_out, out, torch.as_strided(out, (N, M), (1, N)), sum_out
