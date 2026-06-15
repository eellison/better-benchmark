"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 token-plus-position embedding LayerNorm inference scope in one Triton row kernel for the Blenderbot/Pegasus points, including token embedding gather, generated iota position gather, observable bf16 embedding-add output, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)` over the rounded bf16 add, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and the three aliasing flat view returns, whereas Inductor lowers the embedding/add/reduction/affine/cast/view graph through generic embedding, pointwise, and normalization schedules; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization scheduler does not sink indexed embedding producers and an observable pre-norm bf16 materialization into the row-reduction template while preserving repeated alias-only view outputs; the fix is NEW_PATTERN: add a guarded Blenderbot/Pegasus embedding-LayerNorm lowering that fuses row-local token and position gathers into the normalization kernel, keeps the bf16 add boundary exact, and writes the viewed bf16 epilogue directly."""

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
def _embedding_layernorm_h2560_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    EPSILON: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_mask = rows < ROWS
    cols0 = tl.arange(0, 2048)
    cols1 = tl.arange(0, 512) + 2048
    offsets0 = rows[:, None] * 2560 + cols0[None, :]
    offsets1 = rows[:, None] * 2560 + cols1[None, :]

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)[:, None]
    position_id = (rows % SEQ_LEN)[:, None]

    token0 = tl.load(
        token_table_ptr + token_id * 2560 + cols0[None, :],
        mask=row_mask[:, None],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token1 = tl.load(
        token_table_ptr + token_id * 2560 + cols1[None, :],
        mask=row_mask[:, None],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position0 = tl.load(
        position_table_ptr + position_id * 2560 + cols0[None, :],
        mask=row_mask[:, None],
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    position1 = tl.load(
        position_table_ptr + position_id * 2560 + cols1[None, :],
        mask=row_mask[:, None],
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    add0 = _f32_add(token0, position0).to(tl.bfloat16)
    add1 = _f32_add(token1, position1).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets0, add0, mask=row_mask[:, None])
    tl.store(add_out_ptr + offsets1, add1, mask=row_mask[:, None])

    x0 = add0.to(tl.float32)
    x1 = add1.to(tl.float32)
    mean = _f32_mul(_f32_add(tl.sum(x0, axis=1), tl.sum(x1, axis=1)), 1.0 / 2560.0)

    centered0 = _f32_sub(x0, mean[:, None])
    centered1 = _f32_sub(x1, mean[:, None])
    variance = _f32_mul(
        _f32_add(
            tl.sum(_f32_mul(centered0, centered0), axis=1),
            tl.sum(_f32_mul(centered1, centered1), axis=1),
        ),
        1.0 / 2560.0,
    )
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    weight0 = tl.load(weight_ptr + cols0, eviction_policy="evict_last").to(tl.float32)
    weight1 = tl.load(weight_ptr + cols1, eviction_policy="evict_last").to(tl.float32)
    bias0 = tl.load(bias_ptr + cols0, eviction_policy="evict_last").to(tl.float32)
    bias1 = tl.load(bias_ptr + cols1, eviction_policy="evict_last").to(tl.float32)

    norm0 = _f32_mul(centered0, invstd[:, None])
    norm1 = _f32_mul(centered1, invstd[:, None])
    affine0 = _f32_add(_f32_mul(norm0, weight0[None, :]), bias0[None, :])
    affine1 = _f32_add(_f32_mul(norm1, weight1[None, :]), bias1[None, :])
    tl.store(norm_out_ptr + offsets0, affine0.to(tl.bfloat16), mask=row_mask[:, None])
    tl.store(norm_out_ptr + offsets1, affine1.to(tl.bfloat16), mask=row_mask[:, None])


@triton.jit
def _embedding_layernorm_block_kernel(
    token_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)[:, None]
    position_id = (rows % SEQ_LEN)[:, None]

    token = tl.load(
        token_table_ptr + token_id * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    add = _f32_add(token, position).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add, mask=mask)

    x = add.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    norm = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(norm, weight[None, :]), bias[None, :])
    tl.store(norm_out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _run_h2560(inputs, *, ROW_BLOCK: int, num_warps: int, num_stages: int):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_h2560_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        add_out,
        norm_out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        EPSILON=EPS,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        add_out,
        norm_out.view(_shape_tuple(shape0)),
        norm_out.view(_shape_tuple(shape1)),
        norm_out.view(_shape_tuple(shape2)),
    )


def _run_block(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    _embedding_layernorm_block_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        add_out,
        norm_out,
        ROWS=rows,
        SEQ_LEN=seq_len,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        add_out,
        norm_out.view(_shape_tuple(shape0)),
        norm_out.view(_shape_tuple(shape1)),
        norm_out.view(_shape_tuple(shape2)),
    )


# fa552165: (T([8008,2560], bf16), T([32,128], i64), T([128,2560], bf16), ...)
@oracle_impl(hardware="B200", point="fa552165", ROW_BLOCK=1, num_warps=8, num_stages=3)
# cd33d4f9: (T([8008,2560], bf16), T([16,128], i64), T([128,2560], bf16), ...)
@oracle_impl(hardware="B200", point="cd33d4f9", ROW_BLOCK=1, num_warps=8, num_stages=3)
def oracle_forward_h2560(inputs, *, ROW_BLOCK: int, num_warps: int, num_stages: int):
    return _run_h2560(
        inputs,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# f842bd4d: (T([50265,1024], bf16), T([128,128], i64), T([1024,1024], bf16), ...)
@oracle_impl(hardware="B200", point="f842bd4d", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward_h1024(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    return _run_block(
        inputs,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def oracle_forward(inputs):
    raise RuntimeError("registered oracle_impl dispatch should select a point-specific implementation")
