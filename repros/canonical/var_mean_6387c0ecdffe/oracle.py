"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 Swin channel LayerNorm plus fixed 7x7 window partition/permute/clone/final flatten scope, returning the metadata-only input view alias and writing affine-normalized bf16 rows directly in final window order, whereas Inductor lowers the decomposed view/var_mean/affine/cast/permute/clone/view graph through generic normalization and separate layout materialization schedules; Inductor cannot do this today because its scheduler/codegen pattern library has no Swin window-partition LayerNorm template that sinks the static window-layout store into the fixed-hidden row-normalization schedule while preserving bf16 affine rounding; the fix is NEW_PATTERN: add a guarded Swin window LayerNorm-partition lowering that maps final window rows directly to source spatial rows and emits reduction, affine, bf16 cast, and layout epilogue in one kernel."""

import torch
import triton
import triton.language as tl

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
def _swin_window_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    hidden: tl.constexpr,
    window_h: tl.constexpr,
    window_w: tl.constexpr,
    grid_h: tl.constexpr,
    grid_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = out_rows < rows
    col_mask = cols < hidden
    mask = row_mask & col_mask

    inner_w = out_rows % window_w
    tmp = out_rows // window_w
    inner_h = tmp % window_h
    tmp = tmp // window_h
    window_col = tmp % grid_w
    tmp = tmp // grid_w
    window_row = tmp % grid_h
    batch_idx = tmp // grid_h

    src_h = window_row * window_h + inner_h
    src_w = window_col * window_w + inner_w
    src_rows = batch_idx * height * width + src_h * width + src_w

    x = tl.load(
        x_ptr + src_rows * hidden + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    mean = (tl.sum(tl.where(mask, x, 0.0), axis=1) / hidden)[:, None]
    centered = _f32_sub(x, mean)
    variance = (
        tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1) / hidden
    )[:, None]
    invstd = tl.rsqrt(_f32_add(variance, 1.0e-5))

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
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight)
    y = _f32_add(scaled, bias)

    tl.store(
        out_ptr + out_rows * hidden + cols,
        y.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(hardware="B200", point="6d53c060", ROW_BLOCK=4, num_warps=4)
@oracle_impl(hardware="B200", point="b408f6c8", ROW_BLOCK=4, num_warps=4)
def oracle_forward(inputs, *, ROW_BLOCK: int, num_warps: int):
    x, weight, bias, shape0, shape1, _shape2, _shape3, shape4 = inputs
    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape4 = _shape_tuple(shape4)

    view = torch.ops.aten.view.default(x, shape0)
    batch, height, width, hidden = shape0
    _batch, grid_h, window_h, grid_w, window_w, _hidden = shape1
    rows = shape4[0]
    block_h = _next_power_of_2(hidden)

    clone_base = torch.empty_strided(
        (batch, grid_h, grid_w, window_h, window_w, hidden),
        (
            grid_h * grid_w * window_h * window_w * hidden,
            grid_w * window_h * window_w * hidden,
            window_h * window_w * hidden,
            window_w * hidden,
            hidden,
            1,
        ),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out = torch.ops.aten.view.default(clone_base, shape4)

    _swin_window_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        x,
        weight,
        bias,
        out,
        rows=rows,
        height=height,
        width=width,
        hidden=hidden,
        window_h=window_h,
        window_w=window_w,
        grid_h=grid_h,
        grid_w=grid_w,
        BLOCK_H=block_h,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return view, out
