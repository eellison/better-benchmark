"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet 2x2 stride-2 avg_pool2d producer plus the bf16 batchnorm-inference affine, bf16 cast, and NaN-preserving ReLU consumer in one output-tiled Triton stencil while returning both the pooled bf16 tensor and the final bf16 activation; whereas Inductor schedules the pooling result and downstream broadcast affine/ReLU as generic regions with the pooled activation materialized; Inductor cannot do this today because scheduler fusion does not sink fixed-window pooling producers into per-channel affine consumers while preserving the returned side output and bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a stencil-plus-affine epilogue schedule for avg_pool2d feeding inference normalization with multi-output support."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _avgpool_bn_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    pool_out_ptr,
    relu_out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
    EPS_: tl.constexpr,
):
    n = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    out_linear = tl.program_id(2) * BLOCK_O + tl.arange(0, BLOCK_O)

    oh = out_linear // OW
    ow = out_linear - oh * OW
    channel_mask = channels < C
    out_mask = out_linear < (OH * OW)
    mask = channel_mask[:, None] & out_mask[None, :]

    input_base = (
        n * C * H * W
        + channels[:, None] * H * W
        + (oh[None, :] * 2) * W
        + ow[None, :] * 2
    )

    x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + input_base + W, mask=mask, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + input_base + W + 1, mask=mask, other=0.0).to(
        tl.float32
    )

    pooled = _round_bf16_to_f32((x00 + x01 + x10 + x11) * 0.25)

    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=channel_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + EPS_)
    y = (pooled - mean[:, None]) * inv_std[:, None]
    y = y * weight[:, None] + bias[:, None]
    y = _relu_preserve_nan(_round_bf16_to_f32(y))

    output_offsets = (
        n * C * OH * OW + channels[:, None] * OH * OW + out_linear[None, :]
    )
    tl.store(pool_out_ptr + output_offsets, pooled, mask=mask)
    tl.store(relu_out_ptr + output_offsets, y, mask=mask)


# 793856d7: (T([64,512,14,14], bf16), T([512], bf16), ...)
@oracle_impl(hardware="B200", point="793856d7", BLOCK_C=8, BLOCK_O=64, num_warps=4)
# 66d30643: (T([64,256,28,28], bf16), T([256], bf16), ...)
@oracle_impl(hardware="B200", point="66d30643", BLOCK_C=4, BLOCK_O=128, num_warps=4)
# 3a0fa905: (T([64,128,56,56], bf16), T([128], bf16), ...)
@oracle_impl(hardware="B200", point="3a0fa905", BLOCK_C=2, BLOCK_O=256, num_warps=4)
# 8d5f7d9c: (T([128,88,8,8], bf16), T([88], bf16), ...)
@oracle_impl(hardware="B200", point="8d5f7d9c", BLOCK_C=8, BLOCK_O=16, num_warps=4)
# 31290814: (T([128,80,16,16], bf16), T([80], bf16), ...)
@oracle_impl(hardware="B200", point="31290814", BLOCK_C=8, BLOCK_O=64, num_warps=4)
# ec03f8a7: (T([128,64,32,32], bf16), T([64], bf16), ...)
@oracle_impl(hardware="B200", point="ec03f8a7", BLOCK_C=4, BLOCK_O=128, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_O: int,
    num_warps: int,
):
    x, mean, var, weight, bias = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    out_height = height // 2
    out_width = width // 2
    out_shape = (batch, channels, out_height, out_width)
    out_stride = (channels * out_height * out_width, out_height * out_width, out_width, 1)

    pooled = torch.empty_strided(
        out_shape, out_stride, device=x.device, dtype=torch.bfloat16
    )
    relu = torch.empty_strided(
        out_shape, out_stride, device=x.device, dtype=torch.bfloat16
    )
    grid = (
        batch,
        triton.cdiv(channels, BLOCK_C),
        triton.cdiv(out_height * out_width, BLOCK_O),
    )
    _avgpool_bn_relu_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        pooled,
        relu,
        C=channels,
        H=height,
        W=width,
        OH=out_height,
        OW=out_width,
        BLOCK_C=BLOCK_C,
        BLOCK_O=BLOCK_O,
        EPS_=EPS,
        num_warps=num_warps,
        num_stages=3,
    )
    return pooled, relu
