"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete large-row bf16 softmax reduction scope from Repro.forward, including the bf16-to-fp32 input cast, visible fp32 row amax output, visible fp32 row exp-sum denominator output, bf16 softmax materialization rounding boundary, and final bf16 scalar sum without storing the full `[8192,262144]` softmax, whereas Inductor lowers the cast/amax/sub/exp/sum/div/cast/sum graph as generic reductions and pointwise kernels that materialize and reread row-sized intermediates; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize softmax with sibling row-stat outputs and a final reduced bf16 materialization into a streamed online row template; the fix is NEW_PATTERN: add a guarded softmax-stat lowering that emits online row max/denominator accumulators, recomputes rounded probabilities only for the scalar epilogue, and stores the required side outputs directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _softmax_stats_row_sums_kernel(
    x_ptr,
    amax_out_ptr,
    denom_out_ptr,
    row_sums_ptr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * N_COLS

    row_max = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for block_start in tl.range(0, N_COLS, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        x = tl.load(
            x_ptr + row_start + cols,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        block_max = tl.max(x, axis=0)
        next_max = tl.maximum(row_max, block_max)
        denom = _f32_add(
            denom * libdevice.exp(_f32_sub(row_max, next_max)),
            tl.sum(libdevice.exp(_f32_sub(x, next_max)), axis=0),
        )
        row_max = next_max

    row_sum = tl.full((), 0.0, tl.float32)
    for block_start in tl.range(0, N_COLS, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        x = tl.load(
            x_ptr + row_start + cols,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        shifted = _f32_sub(x, row_max)
        probs = libdevice.exp(shifted) / denom
        probs = probs.to(tl.bfloat16).to(tl.float32)
        row_sum = _f32_add(row_sum, tl.sum(tl.where(mask, probs, 0.0), axis=0))

    tl.store(amax_out_ptr + row, row_max)
    tl.store(denom_out_ptr + row, denom)
    tl.store(row_sums_ptr + row, row_sum)


@triton.jit
def _sum_rows_kernel(
    row_sums_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_M)
    mask = offsets < N_ROWS
    vals = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(out_ptr, total.to(tl.bfloat16))


# 4ac62a08: SoftmaxBackward static input bf16[8192,262144].
@oracle_impl(hardware="B200", point="4ac62a08", BLOCK_N=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    (x,) = inputs
    rows = int(x.shape[0])
    cols = int(x.shape[1])

    amax = torch.empty_strided(
        (rows, 1),
        (1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        (rows, 1),
        (1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    row_sums = torch.empty_strided(
        (rows,),
        (1,),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((), (), device=x.device, dtype=torch.bfloat16)

    _softmax_stats_row_sums_kernel[(rows,)](
        x,
        amax,
        denom,
        row_sums,
        N_COLS=cols,
        BLOCK_N=min(BLOCK_N, triton.next_power_of_2(cols)),
        num_warps=num_warps,
        num_stages=3,
    )
    _sum_rows_kernel[(1,)](
        row_sums,
        out,
        N_ROWS=rows,
        BLOCK_M=triton.next_power_of_2(rows),
        num_warps=8,
        num_stages=3,
    )
    return amax, denom, out
