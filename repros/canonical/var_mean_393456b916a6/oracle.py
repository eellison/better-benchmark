"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the metadata view of the first input, the point-specific residual-add cast boundary feeding population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt affine epilogue, final bf16 cast, and the three returned flattened view aliases from the same normalized output buffer, whereas Inductor lowers the captured view/add/var_mean/affine/view graph through its generic normalization scheduler; Inductor cannot do this today because the norm-template scheduler does not canonicalize the fixed-hidden residual add and repeated alias-only view returns into one full-scope LayerNorm lowering; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to inline same-layout residual adds and emit alias-only view returns from the single normalized storage."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _residual_layernorm_alias_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ADD_IN_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    if ADD_IN_FP32:
        x = flat.to(tl.float32) + residual.to(tl.float32)
    else:
        x = (flat + residual).to(tl.bfloat16).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / HIDDEN
    centered = x - mean
    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(var + 1.0e-12)
    y = (centered * invstd * weight + bias).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="d4701d13", BLOCK_N=4096, ADD_IN_FP32=False, num_warps=8)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, ADD_IN_FP32=False, num_warps=4)
@oracle_impl(hardware="B200", point="107053b2", BLOCK_N=2048, ADD_IN_FP32=True, num_warps=8)
@oracle_impl(hardware="B200", point="82b81ff3", BLOCK_N=1024, ADD_IN_FP32=False, num_warps=4)
@oracle_impl(hardware="B200", point="94a8a62c", BLOCK_N=256, ADD_IN_FP32=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N, ADD_IN_FP32, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    norm_shape = tuple(int(dim) for dim in shape0)

    norm_out = torch.empty_strided(
        norm_shape,
        (norm_shape[1] * norm_shape[2], norm_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_alias_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        norm_out,
        HIDDEN=hidden,
        BLOCK_N=BLOCK_N,
        ADD_IN_FP32=ADD_IN_FP32,
        num_warps=num_warps,
    )
    return (
        norm_out,
        norm_out.view(tuple(int(dim) for dim in shape1)),
        norm_out.view(tuple(int(dim) for dim in shape2)),
        norm_out.view(tuple(int(dim) for dim in shape3)),
    )
