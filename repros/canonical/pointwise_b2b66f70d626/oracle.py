"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT bf16 hidden-dimension affine scope, including the metadata input view and the duplicate aliased flattened returns, with one row/hidden Triton tile that reuses the broadcast scale and bias vectors while writing a single contiguous backing buffer, whereas Inductor lowers the view/mul/add/view graph through its generic flattened pointwise scheduler; Inductor cannot do this today because pointwise codegen does not select a fixed-hidden broadcast-aware row template with explicit single-buffer alias-return construction for this affine chain; the fix is NEW_PATTERN: add a guarded row/hidden pointwise template for contiguous trailing-dimension affine broadcasts that preserves the bf16 output dtype, strides, and aliasing views."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 128


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rowwise_affine_alias_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N_COLS)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    x = tl.load(x_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    product = _mul_rn_f32(x, scale[None, :]).to(tl.bfloat16).to(tl.float32)
    y = _add_rn_f32(product, bias[None, :])
    tl.store(out_ptr + offsets, y)


# bc0fb1fb: (T([32768,128], bf16), T([128], bf16), T([128], bf16), S([256,128,128]), S([32768,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="bc0fb1fb", BLOCK_M=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    base = torch.empty_like(arg0_1)
    grid = (triton.cdiv(ROWS, BLOCK_M),)
    _rowwise_affine_alias_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        base,
        N_COLS=COLS,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
    )
    return (
        torch.ops.aten.view.default(base, _shape_param_1),
        torch.ops.aten.view.default(base, _shape_param_2),
    )
