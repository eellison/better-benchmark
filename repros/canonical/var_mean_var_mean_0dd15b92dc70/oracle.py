"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MBART token-embedding plus generated-position embedding plus two dependent hidden-size-1024 LayerNorm scope in one Triton row kernel, including the captured bf16 embedding-add boundary, fp32 population var_mean reductions with eps=1e-5 libdevice.rsqrt, both bf16 affine cast boundaries, the returned first normalized bf16 tensor, and the three final aliasing `[8192,1024]` views from one final buffer, whereas Inductor lowers the embedding producers, two normalization reductions, intermediate bf16 materialization, and repeated view returns through generic scheduler regions; Inductor cannot do this today because its fixed-hidden LayerNorm scheduler does not sink indexed token/position gathers through two dependent normalizations while preserving the observable intermediate bf16 output and alias-only final views; the fix is SCHEDULER_FUSION: extend the embedding-LayerNorm template to fuse gather/add producers, dependent LayerNorm epilogues, intermediate stores, and final view aliases in one full-scope plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
POSITION_OFFSET = 2


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
def _mbart_dual_layernorm_bf16_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight0_ptr,
    bias0_ptr,
    weight1_ptr,
    bias1_ptr,
    first_out_ptr,
    final_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    POSITION_OFFSET_: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    offsets = rows * HIDDEN + cols

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
    position_id = (rows % SEQ_LEN) + POSITION_OFFSET_
    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    x0 = _f32_add(token, position).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    x0_masked = tl.where(mask, x0, 0.0)
    mean0 = _f32_mul(tl.sum(x0_masked, axis=1), 1.0 / HIDDEN)
    centered0 = _f32_sub(x0, mean0[:, None])
    var0 = _f32_mul(
        tl.sum(tl.where(mask, _f32_mul(centered0, centered0), 0.0), axis=1),
        1.0 / HIDDEN,
    )
    invstd0 = libdevice.rsqrt(_f32_add(var0, EPSILON))

    weight0 = tl.load(weight0_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias0 = tl.load(bias0_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    norm0 = _f32_mul(centered0, invstd0[:, None])
    affine0 = _f32_add(_f32_mul(norm0, weight0), bias0)
    first_bf16 = affine0.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(first_out_ptr + offsets, first_bf16, mask=mask)

    x1 = first_bf16.to(tl.float32)
    x1_masked = tl.where(mask, x1, 0.0)
    mean1 = _f32_mul(tl.sum(x1_masked, axis=1), 1.0 / HIDDEN)
    centered1 = _f32_sub(x1, mean1[:, None])
    var1 = _f32_mul(
        tl.sum(tl.where(mask, _f32_mul(centered1, centered1), 0.0), axis=1),
        1.0 / HIDDEN,
    )
    invstd1 = libdevice.rsqrt(_f32_add(var1, EPSILON))

    weight1 = tl.load(weight1_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias1 = tl.load(bias1_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    norm1 = _f32_mul(centered1, invstd1[:, None])
    affine1 = _f32_add(_f32_mul(norm1, weight1), bias1)
    tl.store(
        final_out_ptr + offsets,
        affine1.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


@oracle_impl(hardware="B200", point="7a83b18a", BLOCK_M=1, BLOCK_N=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    (
        token_table,
        token_ids,
        position_table,
        weight0,
        bias0,
        weight1,
        bias1,
        shape0,
        shape1,
        shape2,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len
    first_shape = (batch, seq_len, hidden)
    first_stride = (seq_len * hidden, hidden, 1)

    first = torch.empty_strided(
        first_shape,
        first_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    final = torch.empty_strided(
        first_shape,
        first_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _mbart_dual_layernorm_bf16_kernel[(triton.cdiv(rows, BLOCK_M),)](
        token_table,
        token_ids,
        position_table,
        weight0,
        bias0,
        weight1,
        bias1,
        first,
        final,
        ROWS=rows,
        HIDDEN=hidden,
        SEQ_LEN=seq_len,
        POSITION_OFFSET_=POSITION_OFFSET,
        EPSILON=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        first,
        final.view(tuple(int(dim) for dim in shape0)),
        final.view(tuple(int(dim) for dim in shape1)),
        final.view(tuple(int(dim) for dim in shape2)),
    )
