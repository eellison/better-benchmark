"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 ResNet inference BatchNorm-affine residual-ReLU spatial mean directly into the final `[N,C]` bf16 view by folding the channelwise BatchNorm parameters to scale/shift inside each output row while preserving the explicit bf16 affine cast, bf16 residual add, NaN-preserving ReLU, and fp32 accumulation of the rounded activation, whereas Inductor lowers the broadcast normalization, residual pointwise epilogue, and fixed 7x7/8x8 mean through a generic fused-reduction schedule with repeated per-element normalization algebra; Inductor cannot do this today because its scheduler/codegen does not canonicalize inference-BN parameters into reusable affine scale/shift values for activation-fed spatial mean reductions while preserving dtype boundaries; the fix is ALGEBRAIC_ELIMINATION: canonicalize inference BatchNorm to per-channel scale/shift before lowering the fixed-spatial residual activation mean."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bn_residual_relu_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    spatial = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    spatial_mask = spatial < HW

    n = rows // CHANNELS
    c = rows - n * CHANNELS
    mask = row_mask[:, None] & spatial_mask[None, :]

    element_offsets = rows[:, None] * HW + spatial[None, :]
    x = tl.load(x_ptr + element_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + element_offsets, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

    invstd = (1.0 / tl.sqrt(var + 1.0e-5)) * 1.0
    scale = invstd * weight
    shift = bias - mean * scale
    shifted = x * scale[:, None] + shift[:, None]
    affine_bf16 = shifted.to(tl.bfloat16, fp_downcast_rounding="rtne")
    added_bf16 = (affine_bf16 + residual).to(tl.bfloat16, fp_downcast_rounding="rtne")

    zero_bf16 = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.float32).to(tl.bfloat16)
    relu_bf16 = tl.where(added_bf16 < zero_bf16, zero_bf16, added_bf16)
    relu_f32 = tl.where(spatial_mask[None, :], relu_bf16.to(tl.float32), 0.0)
    reduced = tl.sum(relu_f32, axis=1) / (HW + 0.0)

    tl.store(
        out_ptr + rows,
        reduced.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=row_mask,
    )


def _launch(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    mean, x, var, weight, bias, residual, shape = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    out_shape = tuple(int(dim) for dim in shape)
    output = torch.empty_strided(
        out_shape,
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    total_rows = batch * channels
    _bn_residual_relu_mean_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        output,
        TOTAL_ROWS=total_rows,
        CHANNELS=channels,
        HW=height * width,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


# 187a2d53: resnet18 infer, bf16 [8,512,7,7] residual BN-ReLU mean.
@oracle_impl(hardware="B200", point="187a2d53", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=1)
# 08581e62: resnet50 infer, bf16 [32,2048,7,7] residual BN-ReLU mean.
@oracle_impl(hardware="B200", point="08581e62", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=1)
# 7111f2e4: resnext50_32x4d infer, bf16 [8,2048,7,7] residual BN-ReLU mean.
@oracle_impl(hardware="B200", point="7111f2e4", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=1)
# e2161da4: phlippe_resnet infer, bf16 [128,64,8,8] residual BN-ReLU mean.
@oracle_impl(hardware="B200", point="e2161da4", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=1)
# 2f5dac5e: resnet152 infer, bf16 [128,2048,7,7] residual BN-ReLU mean.
@oracle_impl(hardware="B200", point="2f5dac5e", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=1)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    return _launch(inputs, BLOCK_ROWS=BLOCK_ROWS, BLOCK_HW=BLOCK_HW, num_warps=num_warps)
