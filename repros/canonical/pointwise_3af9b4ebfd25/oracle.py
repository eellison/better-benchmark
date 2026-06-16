"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT bf16 residual-plus-affine pointwise scope in one row/hidden Triton tile, including the required fp32 side output, the final fp32 multiply/add, explicit bf16 output rounding, and both metadata-only views, whereas Inductor lowers the same decomposed view/add/mul/add/cast/view graph through its generic flattened pointwise schedule; Inductor cannot do this today because pointwise codegen does not select a broadcast-aware two-dimensional row template that hoists the hidden-size scale and bias loads while writing sibling outputs with different dtypes; the fix is NEW_PATTERN: add a guarded row/hidden affine pointwise template for contiguous trailing-dimension broadcasts with side-output stores and benchmark-gate it against generic pointwise scheduling."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32768


@triton.jit
def _residual_affine_kernel(
    x_ptr,
    residual_ptr,
    scale_ptr,
    bias_ptr,
    add_out_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    offsets = rows[:, None] * n_cols + cols[None, :]

    x = tl.load(x_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)

    add = x + residual
    y = add * scale[None, :] + bias[None, :]
    tl.store(add_out_ptr + offsets, add)
    tl.store(out_ptr + offsets, y)


# 4c7d1afa: (T([32768,512], bf16), T([256,128,512], f32), T([512], f32), T([512], f32), S([256,128,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="4c7d1afa", block_m=16, block_n=512, num_warps=4)
# b5045c46: (T([32768,128], bf16), T([256,128,128], f32), T([128], f32), T([128], f32), S([256,128,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="b5045c46", block_m=32, block_n=128, num_warps=4)
def oracle_forward(inputs, *, block_m, block_n, num_warps):
    x, residual, scale, bias, _shape0, shape1 = inputs
    n_cols = int(scale.numel())
    add_out = torch.empty_strided(
        tuple(residual.shape),
        tuple(residual.stride()),
        device=residual.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(shape1),
        (n_cols, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(ROWS, block_m),)
    _residual_affine_kernel[grid](
        x,
        residual,
        scale,
        bias,
        add_out,
        out,
        n_cols,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    )
    return add_out, out
