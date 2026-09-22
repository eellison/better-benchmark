"""cuTile port of sum_mean_1db528935b36: ConvNeXtV2 exact-GELU + GRN forward.

Mirrors the Triton oracle's 3 kernels:
  1. Spatial norm: sqrt(sum_hw(gelu_bf16(x)^2)) per (n, c). GELU uses
     in-kernel Abramowitz-Stegun 7.1.26 erf approximation.
  2. Channel mean: sum_c(norm) / C + eps, then ratio = norm / denom.
  3. GRN output: gelu_bf16 * (1 + weight * ratio) + bias, cast to bf16
     (recomputed from x with the same in-kernel erf).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _spatial_norm_kernel(
    x_ptr,        # bf16 NHWC-viewed (N, HW, C)
    norm_ptr,     # f32 (N, C)
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)

    x = ct.load(x_ptr, index=(n, 0, c_block),
                shape=(1, BLOCK_HW, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    # In-kernel Abramowitz-Stegun 7.1.26 erf approximation.
    erf_arg = x_f * 0.7071067811865476
    zero_f = ct.zeros((1, BLOCK_HW, BLOCK_C), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((1, BLOCK_HW, BLOCK_C), 1.0, dtype=ct.float32),
                    ct.full((1, BLOCK_HW, BLOCK_C), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_f = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    gelu = (x_f * 0.5) * (erf_f + 1.0)
    gelu_bf = ct.astype(gelu, ct.bfloat16)
    gelu_f = ct.astype(gelu_bf, ct.float32)

    # Mask HW-OOB positions before sumsq
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_valid = ct.reshape(hw_idx < HW_, (1, BLOCK_HW, 1))
    gelu_f = ct.where(hw_valid, gelu_f, 0.0)

    sumsq = ct.sum(gelu_f * gelu_f, axis=1)  # (1, BLOCK_C)
    result = ct.sqrt(sumsq)
    ct.store(norm_ptr, index=(n, c_block), tile=result)


@ct.kernel
def _channel_mean_kernel(
    norm_ptr,     # f32 (N, C)
    denom_ptr,    # f32 (N,)
    ratio_ptr,    # f32 (N, C)
    C_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_C: ct.Constant[float],
):
    n = ct.bid(0)
    norms = ct.load(norm_ptr, index=(n, 0), shape=(1, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    valid = ct.reshape(c_idx < C_, (1, BLOCK_C))
    norms_masked = ct.where(valid, norms, 0.0)

    total = ct.sum(norms_masked)
    denom = total * INV_C + EPS
    ct.store(denom_ptr, index=(n,), tile=ct.reshape(denom, (1,)))

    denom_2d = ct.reshape(denom, (1, 1))
    ratio = norms / denom_2d
    ct.store(ratio_ptr, index=(n, 0), tile=ratio)


@ct.kernel
def _grn_output_kernel(
    x_ptr,        # bf16 flat NHWC (numel,)
    bias_ptr,     # f32 (C,)
    weight_ptr,   # f32 (C,)
    ratio_ptr,    # f32 (N, C)
    out_ptr,      # bf16 flat NHWC (numel,)
    TOTAL: ct.Constant[int],
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)

    # Recompute GELU (with in-kernel erf).
    erf_arg = x_f * 0.7071067811865476
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((BLOCK,), 1.0, dtype=ct.float32),
                    ct.full((BLOCK,), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_f = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))
    gelu = (x_f * 0.5) * (erf_f + 1.0)
    gelu_bf = ct.astype(gelu, ct.bfloat16)
    gelu_f = ct.astype(gelu_bf, ct.float32)

    # Element index -> (channel, n) computation. NHWC layout means:
    #   linear = n*(HW*C) + hw*C + c ; so channel = linear % C, and
    #   n = linear // (HW*C).
    linear = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    channel = linear % C_
    n_idx = linear // (C_ * HW_)

    bias = ct.astype(ct.gather(bias_ptr, channel), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, channel), ct.float32)
    ratio_flat_idx = n_idx * C_ + channel
    ratio = ct.astype(ct.gather(ratio_ptr, ratio_flat_idx), ct.float32)

    mul3 = gelu_f * ratio
    addcmul = bias + weight * mul3
    out = gelu_f + addcmul
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="3c27f190", BLOCK_HW=64, BLOCK_C=64, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="09211248", BLOCK_HW=256, BLOCK_C=32, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="e3b8317b", BLOCK_HW=1024, BLOCK_C=4, OUT_BLOCK=1024)
@oracle_impl(hardware="B200", point="ec4758af", BLOCK_HW=4096, BLOCK_C=4, OUT_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_HW, BLOCK_C, OUT_BLOCK):
    arg0_1, arg1_1, arg2_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    total = arg0_1.numel()
    device = arg0_1.device
    c_padded = _next_pow2(c)

    # View arg0_1 (channels-last physical, i.e., NHWC contiguous in memory) as
    # (N, HW, C) contiguous — metadata-only via as_strided.
    x_nhwc = arg0_1.as_strided((n, hw, c), (hw * c, c, 1))
    x_flat = arg0_1.as_strided((total,), (1,))

    norm = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1),
                               device=device, dtype=torch.float32)
    denom = torch.empty_strided((n, 1, 1, 1), (1, 1, 1, 1),
                                device=device, dtype=torch.float32)
    ratio = torch.empty_strided((n, c, 1, 1), (c, 1, 1, 1),
                                device=device, dtype=torch.float32)
    out = torch.empty_strided(tuple(arg0_1.shape),
                              (c * h * w, 1, w * c, c),
                              device=device, dtype=torch.bfloat16)
    out_flat = out.as_strided((total,), (1,))

    norm_2d = norm.view(n, c)
    ratio_2d = ratio.view(n, c)
    denom_1d = denom.view(n)

    stream = torch.cuda.current_stream()
    # Kernel 1: spatial norm
    ct.launch(
        stream,
        (n, ct.cdiv(c, BLOCK_C), 1),
        _spatial_norm_kernel,
        (x_nhwc, norm_2d, c, hw, BLOCK_HW, BLOCK_C),
    )
    # Kernel 2: channel mean/ratio
    ct.launch(
        stream,
        (n, 1, 1),
        _channel_mean_kernel,
        (norm_2d, denom_1d, ratio_2d, c, c_padded, 1.0 / c),
    )
    # Kernel 3: grn output — pass ratio as 1D flat
    ratio_1d = ratio.view(n * c)
    ct.launch(
        stream,
        (ct.cdiv(total, OUT_BLOCK), 1, 1),
        _grn_output_kernel,
        (x_flat, arg1_1, arg2_1, ratio_1d, out_flat, total, c, hw, OUT_BLOCK),
    )

    return norm, denom, out
