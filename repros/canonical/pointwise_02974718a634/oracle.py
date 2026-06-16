"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT bf16 residual affine pointwise scope with one row/hidden Triton tile that reuses the broadcast scale and bias vectors while storing both f32 intermediates and the final bf16 view, whereas Inductor lowers the view/add/mul/add/cast/view graph through generic flattened pointwise code for each returned tensor; Inductor cannot do this today because its pointwise scheduler/codegen does not recognize fixed-hidden broadcast affine chains with live intermediate outputs as a row-tiled template with explicit parameter-vector reuse and the required bf16 cast boundary; the fix is NEW_PATTERN: add a guarded row/hidden pointwise template for contiguous trailing-dimension affine broadcasts that preserves returned intermediates, output strides, and the final bf16 view."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_affine_three_output_kernel(
    flat_bf16_ptr,
    residual_ptr,
    scale_ptr,
    bias_ptr,
    add_out_ptr,
    affine_out_ptr,
    bf16_out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = cols[None, :] < N_COLS
    offsets = rows[:, None] * N_COLS + cols[None, :]

    flat = tl.load(flat_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=cols < N_COLS, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < N_COLS, other=0.0).to(tl.float32)

    add_value = flat + residual
    affine_value = add_value * scale[None, :]
    affine_value = affine_value + bias[None, :]
    bf16_value = affine_value.to(tl.bfloat16)

    tl.store(add_out_ptr + offsets, add_value, mask=mask)
    tl.store(affine_out_ptr + offsets, affine_value, mask=mask)
    tl.store(bf16_out_ptr + offsets, bf16_value, mask=mask)


@oracle_impl(hardware="B200", point="b5045c46", BLOCK_M=8, BLOCK_N=128, num_warps=4)
@oracle_impl(hardware="B200", point="4c7d1afa", BLOCK_M=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    cols = int(arg0_1.shape[1])
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    final_shape = tuple(int(dim) for dim in _shape_param_1)

    add_out = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    affine_out = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        final_shape,
        (final_shape[1], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, BLOCK_M),)
    _residual_affine_three_output_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        affine_out,
        bf16_out,
        N_COLS=cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return add_out, affine_out, bf16_out
