"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MobileBERT affine-residual pointwise scope in one row/hidden Triton kernel, including both metadata-only `[32768,128] -> [256,128,128]` input views, separate fp32 mul/add rounding boundaries, the returned fp32 `[256,128,128]` tensor, and the sibling bf16 `[32768,128]` view, whereas Inductor lowers the decomposed view/mul/add/add/mul/add/cast/view graph through its generic multi-output pointwise scheduler; Inductor cannot do this today because pointwise codegen does not select a broadcast-aware row/hidden affine-chain template that reuses the four `[128]` parameter vectors while emitting the fp32 and bf16 sibling outputs from one producer tile; the fix is NEW_PATTERN: add a guarded row-wise affine-chain pointwise template for contiguous hidden-dimension broadcasts with multi-output fp32/bf16 stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32768
COLS = 128


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
def _affine_residual_dual_output_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N_COLS)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    arg0 = tl.load(arg0_ptr + offsets).to(tl.float32)
    arg3 = tl.load(arg3_ptr + offsets).to(tl.float32)
    arg1 = tl.load(arg1_ptr + cols).to(tl.float32)[None, :]
    arg2 = tl.load(arg2_ptr + cols).to(tl.float32)[None, :]
    arg4 = tl.load(arg4_ptr + cols).to(tl.float32)[None, :]
    arg5 = tl.load(arg5_ptr + cols).to(tl.float32)[None, :]

    mul = _f32_mul(arg0, arg1)
    add = _f32_add(mul, arg2)
    add_1 = _f32_add(arg3, add)
    mul_1 = _f32_mul(add_1, arg4)
    add_2 = _f32_add(mul_1, arg5)

    tl.store(out_f32_ptr + offsets, add_2)
    tl.store(out_bf16_ptr + offsets, add_2.to(tl.bfloat16, fp_downcast_rounding="rtne"))


@oracle_impl(hardware="B200", point="8fcb116e", BLOCK_M=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    view_shape = tuple(int(dim) for dim in _shape_param_0)
    view_stride = (view_shape[1] * view_shape[2], view_shape[2], 1)
    out_f32 = torch.empty_strided(
        view_shape,
        view_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16_base = torch.empty_strided(
        view_shape,
        view_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(ROWS, BLOCK_M),)
    _affine_residual_dual_output_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out_f32,
        out_bf16_base,
        N_COLS=COLS,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
    )
    return out_f32, out_bf16_base.view(tuple(int(dim) for dim in _shape_param_2))
