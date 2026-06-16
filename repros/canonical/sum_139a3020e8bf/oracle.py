"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle materializes the returned bf16 GELU-derivative product and accumulates row-block column partials from that bf16-rounded storage in the same Triton pass, whereas Inductor materializes the pointwise tensor and rereads it through a separate reduction; Inductor cannot do this today because its scheduler cannot coordinate a live returned side output, its transposed view alias, and split-row reduction partials while preserving the explicit bf16 rounding before the sum; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to emit a fused materialize-plus-partial-reduction producer for returned tensors with sibling reductions."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _materialize_partial_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    partial_ptr,
    sum_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE_DIRECT_SUM: tl.constexpr,
):
    row_group = tl.program_id(0)
    col_group = tl.program_id(1)

    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    offsets = rows * N + cols

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    rsqrt2 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    one = tl.full([1, 1], 1.0, tl.float32)
    half = tl.full([1, 1], 0.5, tl.float32)
    neg_half = tl.full([1, 1], -0.5, tl.float32)
    rsqrt2pi = tl.full([1, 1], 0.3989422804014327, tl.float32)

    erf_arg = rhs * rsqrt2
    cdf = (libdevice.erf(erf_arg) + one) * half
    exp_arg = (rhs * rhs) * neg_half
    pdf_term = rhs * (libdevice.exp(exp_arg) * rsqrt2pi)
    value = lhs * (cdf + pdf_term)
    value_bf16 = value.to(tl.bfloat16)

    tl.store(out_ptr + offsets, value_bf16, mask=mask)

    partial = tl.sum(tl.where(mask, value_bf16.to(tl.float32), 0.0), axis=0)
    out_cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = out_cols < N
    if STORE_DIRECT_SUM:
        rounded = partial.to(tl.bfloat16).to(tl.float32)
        tl.store(sum_ptr + out_cols, rounded, mask=col_mask)
    else:
        tl.store(partial_ptr + row_group * N + out_cols, partial, mask=col_mask)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_group = tl.program_id(0)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (groups < NUM_GROUPS) & (cols < N)
    values = tl.load(
        partial_ptr + groups * N + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    out_cols = col_group * BLOCK_N + tl.arange(0, BLOCK_N)
    rounded = total.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + out_cols, rounded, mask=out_cols < N)


@oracle_impl(hardware="B200", point="8572c66c", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="9448cee8", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=32, num_warps=8)
@oracle_impl(hardware="B200", point="5c6c31bb", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="3f451298", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="c064ba70", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="67c7a377", ROW_BLOCK=128, BLOCK_N=8, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="8951d544", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="a448a1c6", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="5d96adde", ROW_BLOCK=64, BLOCK_N=32, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="63ec809a", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="05a5e903", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="7c6aa97f", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=32, num_warps=8)
@oracle_impl(hardware="B200", point="58ff2bc5", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="6a0c3b2c", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
@oracle_impl(hardware="B200", point="7c4310e0", ROW_BLOCK=64, BLOCK_N=64, FINAL_BLOCK_N=16, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    num_warps: int,
):
    lhs, rhs, _shape0, _shape1, shape2, shape3 = inputs
    m = int(shape2[0])
    n = int(shape2[1])
    out = torch.empty_strided((m, n), (n, 1), device=lhs.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided(tuple(shape3), (1,), device=lhs.device, dtype=torch.float32)

    num_groups = triton.cdiv(m, ROW_BLOCK)
    direct_sum = num_groups == 1
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty_strided(
            (num_groups, n),
            (n, 1),
            device=lhs.device,
            dtype=torch.float32,
        )

    _materialize_partial_kernel[(num_groups, triton.cdiv(n, BLOCK_N))](
        lhs,
        rhs,
        out,
        partial,
        sum_out,
        M=m,
        N=n,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_N=BLOCK_N,
        STORE_DIRECT_SUM=direct_sum,
        num_warps=num_warps,
    )
    if not direct_sum:
        group_block = 1 << (num_groups - 1).bit_length()
        _final_sum_kernel[(triton.cdiv(n, FINAL_BLOCK_N),)](
            partial,
            sum_out,
            N=n,
            NUM_GROUPS=num_groups,
            GROUP_BLOCK=group_block,
            BLOCK_N=FINAL_BLOCK_N,
            num_warps=8,
        )

    return (out, out.permute(1, 0), sum_out)
