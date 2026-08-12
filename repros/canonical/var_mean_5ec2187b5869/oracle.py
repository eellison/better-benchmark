"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-2 bf16 token-plus-generated-position embedding LayerNorm scope in one Triton row kernel, including the returned bf16 embedding sum, the fp32 population var_mean over the bf16-rounded sum, eps=1e-5 rsqrt, bf16 affine output view, and the generated adjacent-position bool side output as an all-false store, whereas Inductor lowers the decomposed embedding/add/LayerNorm graph through a generic row-reduction schedule and separately materializes the generated position mask; Inductor cannot do this today because the fixed-hidden normalization scheduler does not keep indexed embedding producers resident through the returned bf16 sum, statistics pass, affine epilogue, and algebraically constant sibling side-output store; the fix is SCHEDULER_FUSION: teach the normalization schedule to fuse embedding producers and constant generated-mask epilogues into one full-scope row plan while preserving the bf16 rounding boundary before var_mean."""

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
def _embedding_position_layernorm_mask_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    mask_out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_offsets < ROWS
    col_mask = cols < HIDDEN
    valid = row_mask[:, None] & col_mask[None, :]
    offsets = row_offsets[:, None] * HIDDEN + cols[None, :]

    token_id = tl.load(token_ids_ptr + row_offsets, mask=row_mask, other=0)
    position_id = row_offsets - (row_offsets // SEQ_LEN) * SEQ_LEN

    token = tl.load(
        token_table_ptr + token_id[:, None] * HIDDEN + cols[None, :],
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id[:, None] * HIDDEN + cols[None, :],
        mask=valid,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    add_bf16 = _f32_add(token, position).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=valid)

    x = add_bf16.to(tl.float32)
    x_for_reduce = tl.where(valid, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(valid, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

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
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])
    tl.store(norm_out_ptr + offsets, affine.to(tl.bfloat16), mask=valid)
    tl.store(mask_out_ptr + row_offsets, False, mask=row_mask)


def _as_shape(shape):
    out = []
    known = 1
    unknown = -1
    for index, dim in enumerate(shape):
        value = int(dim)
        out.append(value)
        if value == -1:
            unknown = index
        else:
            known *= value
    if unknown >= 0:
        out[unknown] = 8192 * 768 // known
    return tuple(out)


# 3b387564: (T([50257,768], bf16), T([8,1024], i64), T([1024,768], bf16), T([768], bf16), T([768], bf16), S([8,-1]), S([-1,768]), S([8,1]))
@oracle_impl(hardware="B200", point="3b387564", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, _shape2 = inputs
    batch = int(shape0[0])
    seq_len = int(arg1_1.shape[1])
    hidden = int(arg3_1.shape[0])
    rows = int(arg1_1.numel())
    norm_shape = _as_shape(shape1)

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mask_out = torch.empty_strided(
        (batch, seq_len),
        (seq_len, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    _embedding_position_layernorm_mask_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        add_out,
        norm_out,
        mask_out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out, mask_out
