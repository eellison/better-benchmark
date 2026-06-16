"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GenAI static bf16 RMSNorm forward scope in one Triton row-reduction kernel, including fp32 promotion, per-row mean(square) over hidden size 512, eps=1e-6 rsqrt, fp32 weight multiply, and final bf16 contiguous output, whereas Inductor lowers the decomposed convert/pow/mean/add/rsqrt/mul/mul/convert graph through its generic reduction scheduling path. Inductor cannot do this today because its scheduler/codegen does not consistently select a guarded fixed-hidden RMSNorm row schedule that retains the normalized input tile and affine epilogue as one specialization for this very large row count; the fix is SCHEDULER_FUSION: add a benchmark-gated RMSNorm forward schedule/template for contiguous bf16 rows with fp32 weights that fuses the reduction and affine bf16 store directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-6


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
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
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
    square_sum = tl.sum(tl.where(mask, _f32_mul(x, x), 0.0), axis=1)
    inv_rms = libdevice.rsqrt(_f32_add(square_sum / HIDDEN, EPS_C))

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(x, inv_rms[:, None])
    out = _f32_mul(normalized, weight[None, :])
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# e5ae55b5: GenAI static RMSNormForward bf16[1152000,512], weight f32[512]
@oracle_impl(
    hardware="B200",
    point="e5ae55b5",
    BLOCK_M=8,
    BLOCK_H=512,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    x, weight = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _rmsnorm_forward_kernel[(triton.cdiv(rows, BLOCK_M),)](
        x,
        weight,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
