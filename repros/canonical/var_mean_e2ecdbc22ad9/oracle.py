"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete mixed-dtype residual LayerNorm training scope in one shape-specialized Triton row kernel, including the bf16 flat-input view, fp32 residual add side output, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt, fp32 normalized side output, affine epilogue, final bf16 flattened view, and sibling `rsqrt / 192` output with the captured literal divisor, whereas Inductor lowers the captured add/var_mean/normalized-side-output/affine/view graph through generic normalization and pointwise schedules; Inductor cannot do this today because the norm-template scheduler does not fuse a live mixed-dtype residual producer with multiple required side-output stores and the bf16 final cast in one full-scope row plan; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline same-layout residual adds, retain visible add/normalized/saved-scale outputs, and emit the final cast/view directly from the normalization epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_layernorm_training_kernel(
    flat_bf16_ptr,
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
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = residual + flat
    tl.store(add_out_ptr + offsets, x, mask=mask)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / HIDDEN
    centered = x - mean
    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = tl.rsqrt(var + 1.0e-6)
    norm = centered * invstd
    tl.store(norm_out_ptr + offsets, norm, mask=mask)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    final = norm * weight + bias
    tl.store(final_out_ptr + offsets, final.to(tl.bfloat16), mask=mask)
    tl.store(invstd_div_ptr + row, invstd / 192.0)


# (T([25216, 192], bf16), T([128,197,192], f32), T([192], f32), T([192], f32), S([128,197,192]), S([25216,192]))
@oracle_impl(hardware="B200", point="0ff22f63", BLOCK_H=256, num_warps=1)
# (T([25344, 768], bf16), T([128,198,768], f32), T([768], f32), T([768], f32), S([128,198,768]), S([25344,768]))
@oracle_impl(hardware="B200", point="f13eb73e", BLOCK_H=1024, num_warps=4)
# (T([32768, 768], bf16), T([128,256,768], f32), T([768], f32), T([768], f32), S([128,256,768]), S([32768,768]))
@oracle_impl(hardware="B200", point="7b097b88", BLOCK_H=1024, num_warps=4)
# (T([4096, 2048], bf16), T([32,128,2048], f32), T([2048], f32), T([2048], f32), S([32,128,2048]), S([4096,2048]))
@oracle_impl(hardware="B200", point="5d43e450", BLOCK_H=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    bsz = int(arg1_1.shape[0])
    tokens = int(arg1_1.shape[1])

    add_out = torch.empty_like(arg1_1)
    norm_out = torch.empty_like(arg1_1)
    final_out = torch.empty_like(arg0_1)
    invstd_div = torch.empty(
        (bsz, tokens, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _residual_layernorm_training_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_out,
        final_out,
        invstd_div,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return add_out, norm_out, final_out.view(tuple(_shape_param_1)), invstd_div
