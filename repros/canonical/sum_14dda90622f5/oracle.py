"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Lennard-Jones bf16 producer once, including bf16 input promotion, f32 square/subtract/product arithmetic, the required bf16 materialization, the returned `[16,128]` transpose alias, and the sibling dim-0 sum over the bf16-rounded producer followed by the captured bf16-to-f32 reduction-output round trip. Inductor currently schedules the returned materialization/view and the reduction as separate generic consumers of the same pointwise expression; it cannot do this today because its scheduler does not form one alias-aware producer-store plus reduction plan that preserves an intermediate bf16 rounding boundary for a value that is both returned and reduced. The fix is SCHEDULER_FUSION: teach reduction scheduling to fuse returned bf16 producers with compatible sibling reductions while preserving aliasing and dtype-boundary semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


M = 128
N = 16


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _materialize_alias_and_sum_kernel(
    arg0_ptr,
    arg1_ptr,
    out_ptr,
    sum_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * BLOCK_N + cols

    arg0 = tl.load(arg0_ptr + offsets).to(tl.float32)
    arg1 = tl.load(arg1_ptr + offsets).to(tl.float32)

    squared = _f32_mul(arg1, arg1)
    complement = _f32_sub(tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float32), squared)
    value = _f32_mul(arg0, complement).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, value)
    total = tl.sum(value.to(tl.float32), axis=0)
    rounded_total = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    out_cols = tl.arange(0, BLOCK_N)
    tl.store(sum_ptr + out_cols, rounded_total)


@oracle_impl(hardware="B200", point="387810f3", num_warps=8)
def oracle_forward(inputs, *, num_warps: int):
    arg0, arg1, shape = inputs
    del shape

    out = torch.empty_strided((M, N), (N, 1), device=arg0.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((N,), (1,), device=arg0.device, dtype=torch.float32)
    _materialize_alias_and_sum_kernel[(1,)](
        arg0,
        arg1,
        out,
        sum_out,
        BLOCK_M=M,
        BLOCK_N=N,
        num_warps=num_warps,
        num_stages=1,
    )
    return out.permute(1, 0), sum_out
