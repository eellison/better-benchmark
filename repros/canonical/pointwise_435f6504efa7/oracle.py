"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MobileBERT residual affine pointwise scope with a row/hidden Triton tile that reuses the broadcast scale and bias across several contiguous rows while returning the same final view, whereas Inductor lowers the decomposed view/add/mul/add/view graph as a generic flattened pointwise loop over every element; Inductor cannot do this today because pointwise codegen does not select a broadcast-aware two-dimensional row template for fixed hidden-size affine chains, so it relies on generic linear indexing and per-element broadcast addressing; the fix is NEW_PATTERN: add a guarded row/hidden pointwise template for contiguous last-dimension affine broadcasts and benchmark-gate it against the generic pointwise schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _row_affine_kernel(
    addmm_ptr,
    residual_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, n_cols)
    offsets = rows[:, None] * n_cols + cols[None, :]

    addmm = tl.load(addmm_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    scale = tl.load(scale_ptr + cols).to(tl.float32)[None, :]
    bias = tl.load(bias_ptr + cols).to(tl.float32)[None, :]

    added = _round_bf16_to_fp32(addmm + residual)
    scaled = _round_bf16_to_fp32(added * scale)
    out = _round_bf16_to_fp32(scaled + bias)
    tl.store(out_ptr + offsets, out)


# 09b2e78e: (T([32768,512], bf16), T([256,128,512], bf16), T([512], bf16), T([512], bf16), S([256,128,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="09b2e78e", BLOCK_M=1, num_warps=4)
# d2ddc3c7: (T([32768,128], bf16), T([256,128,128], bf16), T([128], bf16), T([128], bf16), S([256,128,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="d2ddc3c7", BLOCK_M=8, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    n_cols = int(arg0_1.shape[1])
    output = torch.empty_strided(
        tuple(int(dim) for dim in shape0),
        (int(shape0[1]) * n_cols, n_cols, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    grid = (arg0_1.shape[0] // BLOCK_M,)
    _row_affine_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        output,
        n_cols,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
    )
    return output.view(tuple(int(dim) for dim in shape1))
