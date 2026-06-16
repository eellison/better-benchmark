"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GenAI RMSNormBackward static scope by keeping each hidden-size-512 row's fp32 mean-square, eps=1e-6 rsqrt side output, fp32 weight multiply, bf16 rounding, and scalar-sum contribution inside Triton reductions, whereas Inductor lowers the decomposed `pow`/`mean`/`rsqrt`/affine/cast/global-sum graph through generic norm and whole-tensor reduction scheduling with a bf16-normalized intermediate; Inductor cannot do this today because the normalization scheduler does not fuse a row-wise RMSNorm template with a dependent whole-tensor reduction over the bf16-rounded affine output while also emitting the row `rsqrt` side output; the fix is SCHEDULER_FUSION: teach norm-template lowering to sink scalar sum consumers into the RMSNorm epilogue and emit row-stat side outputs from the same schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _rmsnorm_sum_partials_kernel(
    x_ptr,
    weight_ptr,
    rsqrt_out_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    square_sum = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
    mean_square = square_sum / HIDDEN
    inv_rms = libdevice.rsqrt(mean_square + 1.0e-6)
    tl.store(rsqrt_out_ptr + rows, inv_rms, mask=row_mask)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = x * inv_rms[:, None] * weight[None, :]
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    row_sums = tl.sum(tl.where(mask, rounded, 0.0), axis=1)
    partial = tl.sum(tl.where(row_mask, row_sums, 0.0), axis=0)
    tl.store(partials_ptr + tl.program_id(0), partial)


@triton.jit
def _sum_partials_kernel(
    partials_ptr,
    partials2_ptr,
    N_PARTIALS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N_PARTIALS
    values = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(partials2_ptr + tile, total)


@triton.jit
def _final_sum_kernel(
    partials_ptr,
    out_ptr,
    N_PARTIALS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_PARTIALS
    values = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr, total)


# e5ae55b5: (T([1152000,512], bf16), T([512], f32))
@oracle_impl(
    hardware="B200",
    point="e5ae55b5",
    BLOCK_M=8,
    BLOCK_H=512,
    REDUCE_BLOCK=1024,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    REDUCE_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    x, weight = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    n_partials = triton.cdiv(rows, BLOCK_M)
    n_partials2 = triton.cdiv(n_partials, REDUCE_BLOCK)

    rsqrt = torch.empty_strided(
        (rows, 1),
        (1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    partials = torch.empty((n_partials,), device=x.device, dtype=torch.float32)
    partials2 = torch.empty((n_partials2,), device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.bfloat16)

    _rmsnorm_sum_partials_kernel[(n_partials,)](
        x,
        weight,
        rsqrt,
        partials,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _sum_partials_kernel[(n_partials2,)](
        partials,
        partials2,
        N_PARTIALS=n_partials,
        BLOCK_N=REDUCE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _final_sum_kernel[(1,)](
        partials2,
        out,
        N_PARTIALS=n_partials2,
        BLOCK_N=triton.next_power_of_2(n_partials2),
        num_warps=4,
        num_stages=3,
    )
    return rsqrt, out
