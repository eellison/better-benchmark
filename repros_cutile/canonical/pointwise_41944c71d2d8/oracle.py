"""cuTile port of pointwise_41944c71d2d8: ShuffleNet BN + ReLU + channel-shuffle-cat.

The Triton kernel writes an interleaved [N, 2C, H, W] output where even
channels come from a strided skip tensor and odd channels come from the BN
affine + bf16-cast + NaN-preserving ReLU applied to the conv input.

We view the output as [N, C, 2, H*W] to make the interleave a simple axis:
out[n, c, 0, hw] = skip[n, c, h, w]
out[n, c, 1, hw] = relu(BN(conv[n, c, h, w]))

C=232/116/58 is not power-of-2, so we tile over the "row" axis (N*C) via
grid dim and load one channel row (H*W) per program. H*W = 49/196/784 —
we pad to the next power of 2 with masked reads.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _shufflenet_bn_relu_shuffle_kernel(
    mean_ptr,      # bf16 [C]
    conv_ptr,      # bf16 [N, C, H*W]  (contiguous)
    var_ptr,       # bf16 [C]
    weight_ptr,    # bf16 [C]
    bias_ptr,      # bf16 [C]
    skip_ptr,      # bf16 [N, C, H*W]  (contiguous view via .contiguous())
    out_ptr,       # bf16 [N, C, 2, BLOCK_HW]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    # Load per-channel scalars
    mean = ct.astype(
        ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var = ct.astype(
        ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    weight = ct.astype(
        ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias = ct.astype(
        ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)
    inv_std = ct.rsqrt(var + EPS)

    # Load conv/skip rows with zero padding for HW past H*W
    conv_bf16 = ct.load(
        conv_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO)
    conv_f = ct.astype(conv_bf16, ct.float32)
    normalized = (conv_f - mean) * inv_std
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.astype(ct.full((1, 1, BLOCK_HW), 0.0, dtype=ct.float32), ct.bfloat16)
    # NaN-preserving ReLU: keep NaN, else max(x, 0).
    is_nan = affine_bf16 != affine_bf16
    relu_bf16 = ct.where(is_nan, affine_bf16, ct.where(affine_bf16 > zero_bf16, affine_bf16, zero_bf16))

    skip_bf16 = ct.load(
        skip_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO)

    ct.store(out_ptr, index=(n, c, 0, 0), tile=ct.reshape(skip_bf16, (1, 1, 1, BLOCK_HW)))
    ct.store(out_ptr, index=(n, c, 1, 0), tile=ct.reshape(relu_bf16, (1, 1, 1, BLOCK_HW)))


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="46084af7")
@oracle_impl(hardware="B200", point="1e6a9948")
@oracle_impl(hardware="B200", point="ef47f5e1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs
    N = int(arg1_1.shape[0])
    C = int(arg1_1.shape[1])
    H = int(arg1_1.shape[2])
    W = int(arg1_1.shape[3])
    HW = H * W
    BLOCK_HW = _next_pow2(HW)

    conv_flat = arg1_1.reshape(N, C, HW)
    skip_flat = arg5_1.contiguous().reshape(N, C, HW)

    if BLOCK_HW == HW:
        out_padded = torch.empty_strided(
            (N, C, 2, HW), (C * 2 * HW, 2 * HW, HW, 1),
            device=arg1_1.device, dtype=torch.bfloat16,
        )
    else:
        out_padded = torch.empty(
            (N, C, 2, BLOCK_HW),
            device=arg1_1.device, dtype=torch.bfloat16,
        )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C, 1),
        _shufflenet_bn_relu_shuffle_kernel,
        (arg0_1, conv_flat, arg2_1, arg3_1, arg4_1, skip_flat, out_padded, C, HW, BLOCK_HW),
    )

    if BLOCK_HW == HW:
        out = out_padded.view(N, C * 2, H, W).contiguous()
    else:
        out = out_padded[..., :HW].contiguous().view(N, C * 2, H, W)
    return out
