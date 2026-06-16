"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the fp32 square/subtract/product producer once, stores the returned bf16 `[128, 16]` tensor, returns its `[16, 128]` permute alias, and accumulates the sibling dim-0 sum from the bf16-rounded producer values before applying the captured bf16-to-f32 rounding on the reduction output, whereas Inductor schedules the returned materialization/view and the reduction as separate generic consumers of the same pointwise expression; Inductor cannot do this today because its scheduler does not form one multi-output producer-store plus reduction plan that preserves an intermediate bf16 rounding boundary for a value that is both returned and reduced; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse returned bf16 producers with compatible sibling reductions while preserving aliasing and dtype-boundary semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _materialize_and_sum_kernel(
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
    sub = _f32_sub(tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float32), squared)
    value = _f32_mul(arg0, sub)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, value_bf16)
    total = tl.sum(value_bf16.to(tl.float32), axis=0)
    rounded_total = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    out_cols = tl.arange(0, BLOCK_N)
    tl.store(sum_ptr + out_cols, rounded_total)


@oracle_impl(hardware="B200", point="387810f3", num_warps=8)
def oracle_forward(inputs, *, num_warps: int):
    arg0, arg1, _shape = inputs
    out = torch.empty_strided((128, 16), (16, 1), device=arg0.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((16,), (1,), device=arg0.device, dtype=torch.float32)
    _materialize_and_sum_kernel[(1,)](
        arg0,
        arg1,
        out,
        sum_out,
        BLOCK_M=128,
        BLOCK_N=16,
        num_warps=num_warps,
        num_stages=1,
    )
    return out, out.permute(1, 0), sum_out
