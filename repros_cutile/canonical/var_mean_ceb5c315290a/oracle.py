"""cuTile port of var_mean_ceb5c315290a (COOPERATIVE_SPLIT_K): TF-EfficientNet
BN training + SiLU + pad.

Compute BN training statistics + running mean/var updates + affine + bf16 +
SiLU + right/bottom zero pad. Uses torch for the running-mean/var reduction
(which mutates arg1_1/arg2_1 in-place via copy_) and a cuTile kernel for the
per-element SiLU affine + pad epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.00000996502277


@ct.kernel
def _silu_pad_kernel(
    x_ptr,        # bf16 [TOTAL_IN]  (channels-last: c = offset % C)
    mean_ptr,     # f32 [C]
    invstd_ptr,   # f32 [C]
    weight_ptr,   # f32 [C]
    bias_ptr,     # f32 [C]
    out_ptr,      # bf16 [TOTAL_OUT]  (channels-last: c = offset % C)
    TOTAL_IN: ct.Constant[int],
    TOTAL_OUT: ct.Constant[int],
    C: ct.Constant[int],
    IN_H: ct.Constant[int],
    IN_W: ct.Constant[int],
    OUT_W: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    valid_out = offsets < TOTAL_OUT

    # In channels-last: linear offset in the output tensor is
    # n * (C*out_h*out_w) + h * (out_w * C) + w * C + c
    # We need c, w, h, n from offsets.
    c = offsets - (offsets // C) * C
    spatial = offsets // C
    w = spatial - (spatial // OUT_W) * OUT_W
    tmp = spatial // OUT_W
    h = tmp - (tmp // (TOTAL_OUT // (OUT_W * C))) * (TOTAL_OUT // (OUT_W * C))  # oh
    # simpler: TOTAL_OUT = N * C * OUT_H * OUT_W, so:
    # spatial = TOTAL_OUT_H*OUT_W stuff. Use OUT_H = IN_H + 1.
    # Actually: n * (out_h * out_w) + h * out_w + w = spatial
    OUT_H = IN_H + 1
    n = spatial // (OUT_H * OUT_W)
    hw = spatial - n * (OUT_H * OUT_W)
    h = hw // OUT_W
    w = hw - h * OUT_W

    # For valid input positions: h < IN_H and w < IN_W
    valid_input = valid_out & (h < IN_H) & (w < IN_W)

    # Input offset in channels-last: n * (C*IN_H*IN_W) + h*(IN_W*C) + w*C + c
    input_offset = n * (C * IN_H * IN_W) + h * (IN_W * C) + w * C + c
    # For invalid, use offset 0 (will be masked in scatter)
    zero_offset = ct.full(shape=(BLOCK,), fill_value=0, dtype=ct.int32)
    safe_offset = ct.where(valid_input, input_offset, zero_offset)

    x = ct.astype(ct.gather(x_ptr, (safe_offset,)), ct.float32)
    mean = ct.astype(ct.gather(mean_ptr, (c,)), ct.float32)
    invstd = ct.gather(invstd_ptr, (c,))
    weight = ct.astype(ct.gather(weight_ptr, (c,)), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, (c,)), ct.float32)

    norm = (x - mean) * invstd
    affine = norm * weight + bias
    rounded_bf = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf, ct.float32)
    # SiLU: rounded / (exp(-rounded) + 1)
    denom = ct.exp(-rounded_f) + 1.0
    silu = rounded_f / denom
    silu_bf = ct.astype(silu, ct.bfloat16)
    zero_bf = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    out_val = ct.where(valid_input, silu_bf, zero_bf)
    ct.scatter(out_ptr, (offsets,), ct.where(valid_out, out_val, zero_bf), mask=valid_out)


@oracle_impl(hardware="B200", point="8026229d")
@oracle_impl(hardware="B200", point="cdca2f80")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    batch, channels, in_h, in_w = arg0_1.shape
    out_h = in_h + 1
    out_w = in_w + 1
    hw = in_h * in_w
    out_hw = out_h * out_w

    # Compute BN training stats in torch (reduce over [N, H, W])
    x_f32 = arg0_1.to(torch.float32)
    var_mean = torch.ops.aten.var_mean.correction(x_f32, [0, 2, 3], correction=0, keepdim=True)
    variance, row_mean = var_mean[0], var_mean[1]
    add = variance + EPS
    rsqrt = torch.rsqrt(add)  # [1, C, 1, 1]

    # Running stats update (matches Repro's copy_ semantics)
    squeeze = row_mean.squeeze([0, 2, 3])
    mul_1 = squeeze * 0.1
    mul_2 = arg1_1 * 0.9
    add_1 = mul_1 + mul_2

    squeeze_1 = variance.squeeze([0, 2, 3])
    mul_3 = squeeze_1 * RUNNING_VAR_CORRECTION
    mul_4 = mul_3 * 0.1
    mul_5 = arg2_1 * 0.9
    add_2 = mul_4 + mul_5

    # Now compute SiLU + pad via cuTile
    mean_1d = row_mean.view(channels)  # f32 [C]
    invstd_1d = rsqrt.view(channels)   # f32 [C]

    # Output tensor channels-last: [N, C, out_h, out_w] with stride (C*out_h*out_w, 1, out_w*C, C)
    padded = torch.empty_strided(
        (batch, channels, out_h, out_w),
        (channels * out_hw, 1, out_w * channels, channels),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    # Flat view of padded
    total_out = batch * channels * out_hw
    total_in = batch * channels * hw
    # Physical layout: for channels-last, stride[1]=1 means C is innermost.
    # Flat[N*out_h*out_w*C + h*out_w*C + w*C + c] with hw*C+w*C+c = spatial index.
    padded_flat = torch.as_strided(padded, (total_out,), (1,))
    x_flat = torch.as_strided(arg0_1, (total_in,), (1,))

    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(total_out, BLOCK), 1, 1),
        _silu_pad_kernel,
        (x_flat, mean_1d, invstd_1d, arg3_1, arg4_1, padded_flat,
         total_in, total_out, channels, in_h, in_w, out_w, BLOCK),
    )

    # Update running stats via copy_ (matches eager semantics)
    arg1_1.copy_(add_1)
    arg2_1.copy_(add_2)

    return row_mean, rsqrt, padded, arg1_1, arg2_1
