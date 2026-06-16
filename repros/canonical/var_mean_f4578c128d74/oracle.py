"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete returned bf16 residual-add tensor and the fp32 population-variance affine LayerNorm consumer in one Triton row kernel, whereas Inductor currently treats the returned add and norm-template reduction as generic scheduled work with the add value materialized for output before the reduction consumes it; Inductor cannot do this today because its scheduler cannot fuse a live returned producer into fixed-hidden LayerNorm codegen while preserving the producer's bf16 rounding and output scope; the fix is SCHEDULER_FUSION: teach norm-template fusion to store the returned residual-add output and reuse that rounded row value directly for the reduction and affine epilogue."""

import torch
import triton
import triton.language as tl

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
    added_bf16 = (flat + residual).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, added_bf16, mask=mask)

    x = added_bf16.to(tl.float32)
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / HIDDEN
    centered = x - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = tl.rsqrt(variance + 1.0e-6)

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


@oracle_impl(hardware="B200", point="155170ab", BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ad6d6241", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="a2357153", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="f049abfe", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="d8c968d2", BLOCK_N=256, num_warps=4)
@oracle_impl(hardware="B200", point="9801ab6a", BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="c4bf51cc", BLOCK_N=4096, num_warps=8)
@oracle_impl(hardware="B200", point="7f824027", BLOCK_N=4096, num_warps=8)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="17affd46", BLOCK_N=1024, num_warps=8)
@oracle_impl(hardware="B200", point="0b3dc49f", BLOCK_N=1024, num_warps=8)
@oracle_impl(hardware="B200", point="ceab07f0", BLOCK_N=1024, num_warps=8)
@oracle_impl(hardware="B200", point="a3e95c29", BLOCK_N=1024, num_warps=8)
@oracle_impl(hardware="B200", point="f2e11670", BLOCK_N=1024, num_warps=8)
@oracle_impl(hardware="B200", point="324149d9", BLOCK_N=512, num_warps=4)
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
