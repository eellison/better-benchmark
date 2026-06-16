"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete lennard_jones bf16 outer-product affine plus tanh scope with a tiled Triton broadcast kernel, including the f32 input promotions, captured multiply-by-one identities, f32 add, explicit bf16 rounding before `aten.tanh`, and bf16 output store, whereas Inductor lowers the graph as a generic flattened pointwise kernel over the broadcasted `[128, 16]` output; Inductor cannot do this today because pointwise codegen does not select a small outer-broadcast template that hoists invariant row/column loads while preserving dtype-rounding boundaries; the fix is NEW_PATTERN: add a guarded tiled broadcast-pointwise lowering for small outer products with exact bf16 cast placement and transcendental epilogues."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _outer_affine_tanh_kernel(
    rows_ptr,
    cols_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, 16)
    row_mask = row_offsets < ROWS

    row_vals = tl.load(rows_ptr + row_offsets[:, None], mask=row_mask[:, None], other=0.0).to(tl.float32)
    col_vals = tl.load(cols_ptr + col_offsets[None, :]).to(tl.float32)
    bias_vals = tl.load(bias_ptr + col_offsets[None, :]).to(tl.float32)

    product = _f32_mul(row_vals, col_vals)
    product = _f32_mul(product, 1.0)
    bias_vals = _f32_mul(bias_vals, 1.0)
    affine = _f32_add(product, bias_vals)
    rounded = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    result = libdevice.tanh(rounded).to(tl.bfloat16, fp_downcast_rounding="rtne")

    out_offsets = row_offsets[:, None] * COLS + col_offsets[None, :]
    tl.store(out_ptr + out_offsets, result, mask=row_mask[:, None])


# c67d6e69: (T([128,1], bf16), T([16,1], bf16), T([16], bf16))
@oracle_impl(hardware="B200", point="c67d6e69", BLOCK_M=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    arg0_1, arg1_1, arg2_1 = inputs
    rows = 128
    cols = 16
    out = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _outer_affine_tanh_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        ROWS=rows,
        COLS=cols,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
