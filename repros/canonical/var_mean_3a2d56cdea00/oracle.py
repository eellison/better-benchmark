"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete returned bf16 residual-add tensor and fp32 population-variance affine LayerNorm consumer in one Triton row kernel, including the metadata view of the flat input, bf16-rounded add side output, `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, affine multiply/add, and final bf16 view with no tolerance-based resident/strict output selection, whereas Inductor lowers the captured add/var_mean/affine/view graph through its generic normalization scheduler; Inductor cannot do this today because its scheduler cannot fuse a live returned producer into fixed-hidden LayerNorm codegen while preserving the returned producer store and normalization output scope; the fix is SCHEDULER_FUSION: teach norm-template fusion to store the returned residual-add output and feed the graph-faithful bf16-rounded producer into the reduction and affine epilogue."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _residual_layernorm_two_output_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    x = flat + residual
    added_bf16 = x.to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, added_bf16, mask=mask)

    x = added_bf16.to(tl.float32)
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / HIDDEN
    centered = x - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(variance + 1.0e-5)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (centered * invstd * weight + bias).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, y, mask=mask)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    unknown = -1
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            unknown = idx
        else:
            known *= dim
    if unknown >= 0:
        dims[unknown] = numel // known
    return tuple(dims)


# (T([16384,768], bf16), T([32,512,768], bf16), T([768], bf16), T([768], bf16), S([32,512,768]), S([-1,768]))
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, num_warps=4)
# (T([8192,768], bf16), T([8,1024,768], bf16), T([768], bf16), T([768], bf16), S([8,1024,768]), S([-1,768]))
@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_N=1024, num_warps=4)
# (T([4096,2048], bf16), T([32,128,2048], bf16), T([2048], bf16), T([2048], bf16), S([32,128,2048]), S([4096,2048]))
@oracle_impl(hardware="B200", point="f2c837cd", BLOCK_N=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in _shape_param_0)
    norm_shape = _resolve_shape(_shape_param_1, arg0_1.numel())

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_two_output_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        HIDDEN=hidden,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return (add_out, norm_out)
