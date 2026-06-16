"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT bf16 residual affine pointwise scope with one row/hidden Triton tile that reuses the broadcast scale and bias vectors while storing one bf16 backing tensor and returning both the `[256,128,128]` result and its `[32768,128]` view alias, whereas Inductor lowers the view/add/mul/add/view graph through its generic flattened pointwise scheduler; Inductor cannot do this today because pointwise codegen has no guarded fixed-hidden broadcast-affine template that preserves the multi-output view alias while specializing parameter-vector reuse; the fix is NEW_PATTERN: add a row/hidden pointwise template for contiguous trailing-dimension affine broadcasts that emits the required backing tensor once and returns metadata views of it."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bf16_residual_affine_alias_kernel(
    flat_ptr,
    residual_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
    col_mask = cols < N_COLS
    offsets = rows[:, None] * N_COLS + cols[None, :]

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    add_value = (flat + residual).to(tl.bfloat16).to(tl.float32)
    mul_value = (add_value * scale[None, :]).to(tl.bfloat16).to(tl.float32)
    value = (mul_value + bias[None, :]).to(tl.bfloat16)
    tl.store(out_ptr + offsets, value, mask=mask)


# (T([32768,128], bf16), T([256,128,128], bf16), T([128], bf16), T([128], bf16),
#  S([256,128,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="d2ddc3c7", BLOCK_M=8, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    cols = int(arg0_1.shape[1])
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    final_shape = tuple(int(dim) for dim in _shape_param_1)

    output = torch.empty_strided(
        view_shape,
        (view_shape[1] * view_shape[2], view_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, BLOCK_M),)
    _bf16_residual_affine_alias_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        output,
        N_ROWS=rows,
        N_COLS=cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return output, output.view(final_shape)
