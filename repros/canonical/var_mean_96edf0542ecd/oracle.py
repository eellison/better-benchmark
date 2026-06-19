"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the metadata view of the flat input, the captured residual-add/convert boundary, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt affine epilogue with bf16 scale/bias, final bf16 cast, and the returned contiguous flattened view, whereas Inductor lowers the captured view/add/var_mean/affine/view graph through its generic normalization scheduler; Inductor cannot do this today because the norm-template scheduler does not canonicalize the same-layout residual add and final alias-only view into one full-scope fixed-hidden row plan while preserving the observable bf16/fp32 cast-boundary contract; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline residual adds into the row statistics and affine epilogue and emit the requested view layout directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _residual_layernorm_view_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x = (flat + residual).to(tl.bfloat16).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / HIDDEN
    centered = x - mean
    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(var + 1.0e-12)
    y = (centered * invstd * weight + bias).to(tl.bfloat16)

    tl.store(out_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="B200", point="d4701d13", BLOCK_N=4096, num_warps=8)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="107053b2", BLOCK_N=2048, num_warps=8)
@oracle_impl(hardware="B200", point="82b81ff3", BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="94a8a62c", BLOCK_N=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = tuple(int(dim) for dim in _shape_param_1)

    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_view_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        HIDDEN=hidden,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out
