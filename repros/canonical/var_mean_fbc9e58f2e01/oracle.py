"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete OPT bf16 token-plus-position embedding LayerNorm inference scope in one shape-specialized Triton row kernel, including token embedding gather, captured `(position - 1).to(int64) + 2` position gather, observable bf16 embedding-add output, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` over that rounded bf16 add, eps=1e-5 rsqrt, bf16 scale/bias affine epilogue, final bf16 cast, and the three aliasing `[8192,768]` view returns; Inductor lowers the embedding producers, row normalization, affine cast, and repeated view returns through generic embedding and normalization schedules; Inductor cannot do this today because the norm-template scheduler does not fold generated token/position gathers and alias-only view outputs into one fixed-hidden row-normalization plan while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach embedding-LayerNorm scheduling to fold generated position gathers into the fixed-hidden row-normalization template while preserving bf16 rounding and alias-only outputs."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
SEQ_LEN = 2048
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
def _opt_embedding_layernorm_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < 8192
    col_mask = cols < 768
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * 768 + cols[None, :]

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)[:, None]
    position_f32 = tl.load(position_ids_ptr + rows, mask=row_mask, other=0.0)
    position_id = (position_f32 - 1.0).to(tl.int64) + 2

    token = tl.load(
        token_table_ptr + token_id * 768 + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id[:, None] * 768 + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    x_fp32 = _f32_add(token, position)
    x_bf16 = x_fp32.to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, x_bf16, mask=mask)
    x = x_bf16.to(tl.float32)

    x_masked = tl.where(col_mask[None, :], x, 0.0)
    mean = tl.sum(x_masked, axis=1) / 768.0
    centered = _f32_sub(x, mean[:, None])
    variance = tl.sum(
        tl.where(col_mask[None, :], _f32_mul(centered, centered), 0.0),
        axis=1,
    ) / 768.0
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-5))

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    tl.store(norm_out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


@oracle_impl(
    hardware="B200",
    point="801db743",
    BLOCK_M=4,
    BLOCK_H=1024,
    num_warps=8,
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_shape_params = inputs
    add_out = torch.empty_strided(
        (4, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _opt_embedding_layernorm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        add_out,
        norm_out,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out, norm_out, norm_out
