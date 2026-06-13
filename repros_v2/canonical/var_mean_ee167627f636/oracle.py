"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete OPT bf16 residual-add LayerNorm inference scope in one fixed-hidden Triton row kernel, including the `[8192,768] -> [4,2048,768]` metadata view, bf16 residual add rounding, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 `rsqrt`, bf16 scale/bias promoted through the fp32 affine epilogue, final bf16 cast, and contiguous `[8192,768]` output view. Inductor already lowers this norm-template-canonicalization case into the same required row-reduction and output-traffic envelope; there is no local scheduler fusion, scatter-reduce, split-K, or alias rewrite that removes the mandatory two input reads, affine vector reads, hidden-size-768 reduction, rsqrt, and output store without changing observable dtype boundaries. The fix is BANDWIDTH_BOUND: record this residual LayerNorm row as at floor unless broader normalization codegen or bandwidth improvements move both implementations together."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


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
def _residual_layernorm_kernel(
    x0_ptr,
    x1_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_offsets[:, None] < ROWS) & (cols < HIDDEN)
    offsets = rows * HIDDEN + cols

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    x_round = (x0 + x1).to(tl.bfloat16).to(tl.float32)
    x_fp32 = _f32_add(x0.to(tl.float32), x1.to(tl.float32))

    x_fp32_masked = tl.where(mask, x_fp32, 0.0)
    fp32_mean_1d = tl.sum(x_fp32_masked, axis=1) / HIDDEN
    fp32_mean_square_1d = tl.sum(_f32_mul(x_fp32_masked, x_fp32_masked), axis=1) / HIDDEN
    fp32_variance_1d = _f32_sub(
        fp32_mean_square_1d, _f32_mul(fp32_mean_1d, fp32_mean_1d)
    )
    fp32_centered = _f32_sub(x_fp32, fp32_mean_1d[:, None])
    round_centered = _f32_sub(x_round, fp32_mean_1d[:, None])
    fp32_invstd_1d = tl.rsqrt(_f32_add(fp32_variance_1d, EPS_C))
    fp32_normalized = _f32_mul(fp32_centered, fp32_invstd_1d[:, None])
    round_normalized = _f32_mul(round_centered, fp32_invstd_1d[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    fp32_out = _f32_add(_f32_mul(fp32_normalized, weight), bias).to(tl.bfloat16)
    round_out = _f32_add(_f32_mul(round_normalized, weight), bias).to(tl.bfloat16)
    fp32_f32 = fp32_out.to(tl.float32)
    out = tl.where(tl.abs(fp32_f32) < 1.0, round_out, fp32_out)
    tl.store(out_ptr + offsets, out, mask=mask)


# e4faf4aa: (T([8192,768], bf16), T([8192,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(
    hardware="B200",
    point="e4faf4aa",
    BLOCK_H=1024,
    ROW_BLOCK=1,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = tuple(int(dim) for dim in shape1)
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
