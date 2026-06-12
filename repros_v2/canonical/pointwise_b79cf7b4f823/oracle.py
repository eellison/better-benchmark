"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MobileBERT affine-alias scope in one storage-linear Triton pass and returns the materialized `[256, 128, 512]` tensor plus the three `[32768, 512]` metadata views aliasing the same buffer, whereas Inductor lowers the add/mul/add/view graph through its generic pointwise scheduler; Inductor cannot do this today because pointwise codegen has no guarded B200 dense affine-alias specialization for this broadcast-512 shape family that preserves the expanded multi-output alias contract; the fix is NEW_PATTERN: add an alias-aware dense affine pointwise specialization with the same bf16 rounding and view-return semantics when it beats generic pointwise codegen."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _affine_alias_bf16_kernel(
    arg0_ptr,
    arg1_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    cols = offsets % n_cols

    x = tl.load(arg0_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(arg1_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    y = (x + residual) * scale + bias
    tl.store(out_ptr + offsets, y, mask=mask)


# 09b2e78e: (T([32768,512], bf16), T([256,128,512], bf16), T([512], bf16), T([512], bf16), S([256,128,512]), S([32768,512]), S([32768,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="09b2e78e", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs[:4]
    shape_0, shape_1, shape_2, shape_3 = inputs[4:8]

    out = torch.empty_strided(
        tuple(shape_0),
        (shape_0[1] * shape_0[2], shape_0[2], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    n_elements = arg0_1.numel()
    grid = (triton.cdiv(n_elements, BLOCK),)
    _affine_alias_bf16_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        n_elements,
        arg0_1.shape[1],
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return (out, out.view(shape_1), out.view(shape_2), out.view(shape_3))
