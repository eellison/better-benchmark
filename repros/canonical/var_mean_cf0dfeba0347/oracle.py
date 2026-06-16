"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo f32 token embedding gather, generated position-id side output, all-false adjacent-position mask, position embedding gather, f32 token-plus-position sum, hidden-size-2048 population var_mean(..., dim=2, correction=0, keepdim=True), eps=1e-5 rsqrt side output, affine LayerNorm epilogue, and final bf16 [4096,2048] view in one shape-specialized Triton row kernel, whereas Inductor lowers the embedding producers, generated iota/cat/slice/ne mask, row normalization, affine cast, and side-output stores through generic embedding, pointwise, and norm-template schedules; Inductor cannot do this today because norm-template canonicalization does not recognize GPT-Neo generated-position embedding LayerNorm with several live producer/statistic side outputs and a deterministic sibling mask as one semantic pattern; the fix is NEW_PATTERN: add a GPT-style embedding-LayerNorm template that folds token and position gathers, row statistics, affine stores, side-output materialization, and constant-mask emission into one guarded lowering."""

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
def _gptneo_embedding_layernorm_full_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    token_out_ptr,
    position_ids_out_ptr,
    mask_out_ptr,
    position_out_ptr,
    add_out_ptr,
    mean_out_ptr,
    invstd_out_ptr,
    bf16_out_ptr,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN
    seq_index = row % SEQ
    offsets = row * HIDDEN + cols

    token_id = tl.load(token_ids_ptr + row)
    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + seq_index * HIDDEN + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x = _f32_add(token, position)

    tl.store(token_out_ptr + offsets, token, mask=col_mask)
    tl.store(add_out_ptr + offsets, x, mask=col_mask)
    tl.store(mask_out_ptr + row, False)
    if row < SEQ:
        tl.store(position_ids_out_ptr + row, row)
        tl.store(position_out_ptr + row * HIDDEN + cols, position, mask=col_mask)

    x_for_reduce = tl.where(col_mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=0) / HIDDEN
    centered = _f32_sub(x, mean)
    centered_for_var = tl.where(col_mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    y = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias)

    tl.store(mean_out_ptr + row, mean)
    tl.store(invstd_out_ptr + row, invstd)
    tl.store(bf16_out_ptr + offsets, y.to(tl.bfloat16), mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="2b53f84f", BLOCK_H=2048, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    token_table, token_ids, position_table, weight, bias, _expand_shape, view_shape = inputs

    batch = int(token_ids.shape[0])
    seq = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq
    token_out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    position_ids = torch.empty_strided(
        (1, seq),
        (seq, 1),
        device=token_table.device,
        dtype=torch.int64,
    )
    mask_out = torch.empty_strided(
        (batch, seq),
        (seq, 1),
        device=token_table.device,
        dtype=torch.bool,
    )
    position_out = torch.empty_strided(
        (1, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    add_out = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (batch, seq, 1),
        (seq, 1, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (batch, seq, 1),
        (seq, 1, 1),
        device=token_table.device,
        dtype=torch.float32,
    )
    bf16_base = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _gptneo_embedding_layernorm_full_kernel[(rows,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        token_out,
        position_ids,
        mask_out,
        position_out,
        add_out,
        mean,
        invstd,
        bf16_base,
        ROWS=rows,
        SEQ=seq,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        token_out,
        position_ids,
        mask_out,
        position_out,
        add_out,
        mean,
        invstd,
        bf16_base.view(_shape_tuple(view_shape)),
    )
