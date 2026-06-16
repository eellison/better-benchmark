"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet bf16 batchnorm-affine, natural-exp SiLU, explicit bf16 activation materialization, and spatial mean over that bf16 activation with channels-last output strides, whereas Inductor lowers the same scope as a generic pointwise producer followed by a persistent mean reduction; Inductor cannot do this today because its scheduler does not have a guarded BN-affine/SiLU/spatial-mean template that preserves the bf16 rounding boundary and NaN propagation while specializing the channels-last row reduction; the fix is SCHEDULER_FUSION: add a reusable scheduler fusion template for BN-affine plus natural-exp SiLU feeding small spatial means, with the explicit bf16 producer boundary retained when the activation is also returned."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

from oracle_harness import oracle_impl


EPS = 0.001


@triton.jit
def _bn_silu_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    c = offsets % channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

    y = (x - mean) * (1.0 / libdevice.sqrt(var + eps))
    y = y * weight + bias
    y = y / (tl_math.exp(-y) + 1.0)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _spatial_mean_kernel(
    activation_ptr,
    mean_out_ptr,
    rows: tl.constexpr,
    channels: tl.constexpr,
    hw: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row = row_offsets[:, None]
    r = tl.arange(0, BLOCK_R)[None, :]
    mask_row = row < rows
    mask_r = r < hw

    c = row % channels
    n = row // channels
    offsets = n * channels * hw + r * channels + c
    values = tl.load(activation_ptr + offsets, mask=mask_row & mask_r, other=0.0).to(tl.float32)
    values = tl.where(mask_r, values, 0.0)
    reduced = tl.sum(values, axis=1) / hw
    tl.store(mean_out_ptr + row_offsets, reduced, mask=row_offsets < rows)


@oracle_impl(hardware="B200", point="64799fb2", P_BLOCK=1024, M_BLOCK=16, R_BLOCK=64, P_WARPS=4, M_WARPS=4)
@oracle_impl(hardware="B200", point="aa1adfec", P_BLOCK=1024, M_BLOCK=16, R_BLOCK=64, P_WARPS=4, M_WARPS=4)
@oracle_impl(hardware="B200", point="8162a5bc", P_BLOCK=1024, M_BLOCK=8, R_BLOCK=256, P_WARPS=4, M_WARPS=4)
@oracle_impl(hardware="B200", point="d46c34b3", P_BLOCK=1024, M_BLOCK=8, R_BLOCK=256, P_WARPS=4, M_WARPS=4)
@oracle_impl(hardware="B200", point="560aaeca", P_BLOCK=1024, M_BLOCK=8, R_BLOCK=256, P_WARPS=4, M_WARPS=4)
@oracle_impl(hardware="B200", point="9ab2d895", P_BLOCK=1024, M_BLOCK=4, R_BLOCK=1024, P_WARPS=4, M_WARPS=8)
@oracle_impl(hardware="B200", point="ebe204a7", P_BLOCK=1024, M_BLOCK=4, R_BLOCK=1024, P_WARPS=4, M_WARPS=8)
@oracle_impl(hardware="B200", point="51719261", P_BLOCK=1024, M_BLOCK=2, R_BLOCK=4096, P_WARPS=4, M_WARPS=8)
@oracle_impl(hardware="B200", point="d53a7e50", P_BLOCK=1024, M_BLOCK=2, R_BLOCK=4096, P_WARPS=4, M_WARPS=8)
@oracle_impl(hardware="B200", point="9c97edfa", P_BLOCK=1024, M_BLOCK=1, R_BLOCK=16384, P_WARPS=4, M_WARPS=8)
def oracle_forward(
    inputs,
    *,
    P_BLOCK: int,
    M_BLOCK: int,
    R_BLOCK: int,
    P_WARPS: int,
    M_WARPS: int,
):
    mean, x, var, weight, bias = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width

    activation = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    n_elements = activation.numel()
    _bn_silu_kernel[(triton.cdiv(n_elements, P_BLOCK),)](
        mean,
        x,
        var,
        weight,
        bias,
        activation,
        n_elements=n_elements,
        channels=channels,
        eps=EPS,
        BLOCK=P_BLOCK,
        num_warps=P_WARPS,
        num_stages=3,
    )
    rows = batch * channels
    _spatial_mean_kernel[(triton.cdiv(rows, M_BLOCK),)](
        activation,
        mean_out,
        rows=rows,
        channels=channels,
        hw=hw,
        BLOCK_M=M_BLOCK,
        BLOCK_R=R_BLOCK,
        num_warps=M_WARPS,
        num_stages=3,
    )
    return activation, mean_out
