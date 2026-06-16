"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BEiT mixed-dtype affine-residual LayerNorm training scope in one Triton row kernel, including the bf16 flat input view, fp32 gamma multiply before the residual add, returned f32 add tensor, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, returned f32 normalized tensor, affine scale/bias epilogue, final bf16 flattened view, and sibling `rsqrt / 768` output, whereas Inductor lowers the captured producer/var_mean/affine/view graph through generic normalization and pointwise scheduling; Inductor cannot do this today because the norm-template scheduler does not keep the mixed-dtype producer, multiple visible side-output stores, final bf16 cast, and side-scale epilogue in one full-scope row plan; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline bf16-to-fp32 affine residual producers while retaining add/normalized/side outputs and emitting the final cast/view directly from the row kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _mixed_affine_residual_layernorm_kernel(
    flat_bf16_ptr,
    gamma_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    final_out_ptr,
    invstd_div_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_bf16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = residual + gamma * flat
    tl.store(add_out_ptr + offsets, x, mask=mask)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / HIDDEN
    centered = x - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(variance + 1.0e-6)
    norm = centered * invstd
    tl.store(norm_out_ptr + offsets, norm, mask=mask)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    final = norm * weight + bias
    tl.store(final_out_ptr + offsets, final.to(tl.bfloat16), mask=mask)
    tl.store(invstd_div_ptr + row, invstd / HIDDEN)


# f4c82f7a: BEiT mixed bf16/f32 residual LayerNorm, rows=25216, hidden=768.
@oracle_impl(hardware="B200", point="f4c82f7a", BLOCK_H=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in shape0)
    final_shape = tuple(int(dim) for dim in shape1)

    add_out = torch.empty_like(arg2_1)
    norm_out = torch.empty_like(arg2_1)
    final_out = torch.empty_strided(
        final_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (add_shape[0], add_shape[1], 1),
        (add_shape[1], 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _mixed_affine_residual_layernorm_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        add_out,
        norm_out,
        final_out,
        invstd_div,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return add_out, norm_out, final_out.view(final_shape), invstd_div
