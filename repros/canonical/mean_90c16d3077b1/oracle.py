"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 BatchNorm-inference affine, explicit bf16 cast, NaN-preserving ReLU, returned activation tensor, and keepdim spatial mean in one Triton row kernel, whereas Inductor lowers the broadcast affine/ReLU producer and the spatial mean consumer through generic pointwise and reduction scheduling around a materialized activation; Inductor cannot do this today because its scheduler does not sink a returned pointwise producer into a simultaneous returned-output plus reduction epilogue while preserving the bf16 cast boundary before ReLU; the fix is SCHEDULER_FUSION: teach Inductor to fuse BN-affine activation producers with same-tensor returned output stores and spatial mean reductions in one generated kernel."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bn_relu_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    relu_out_ptr,
    mean_out_ptr,
    total_rows: tl.constexpr,
    channels: tl.constexpr,
    hw: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_HW)
    hw_mask = cols < hw
    channel = row - (row // channels) * channels
    offsets = row * hw + cols

    x = tl.load(x_ptr + offsets, mask=hw_mask, other=0.0).to(tl.float32)
    channel_mean = tl.load(mean_ptr + channel).to(tl.float32)
    variance = tl.load(var_ptr + channel).to(tl.float32)
    weight = tl.load(weight_ptr + channel).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)

    invstd = 1.0 / tl.sqrt(variance + 0.001)
    y = ((x - channel_mean) * invstd) * weight + bias
    y_bf16 = y.to(tl.bfloat16, fp_downcast_rounding="rtne")
    zero = tl.full((BLOCK_HW,), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where(y_bf16 < zero, zero, y_bf16)
    tl.store(relu_out_ptr + offsets, relu, mask=hw_mask)

    mean_value = tl.sum(tl.where(hw_mask, relu.to(tl.float32), 0.0), axis=0) / hw
    tl.store(mean_out_ptr + row, mean_value.to(tl.bfloat16, fp_downcast_rounding="rtne"))


def _launch(inputs, *, BLOCK_HW: int, num_warps: int, num_stages: int):
    mean, x, variance, weight, bias = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    total_rows = batch * channels

    relu_out = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    mean_out = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _bn_relu_mean_kernel[(total_rows,)](
        mean,
        x,
        variance,
        weight,
        bias,
        relu_out,
        mean_out,
        total_rows=total_rows,
        channels=channels,
        hw=hw,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (relu_out, mean_out)


# 51b8f364: (T([120], bf16), T([32,120,28,28], bf16), ...)
# e78adaa5: (T([72], bf16), T([32,72,28,28], bf16), ...)
@oracle_impl(hardware="B200", point="51b8f364", BLOCK_HW=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="e78adaa5", BLOCK_HW=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_HW: int, num_warps: int, num_stages: int):
    return _launch(
        inputs,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
