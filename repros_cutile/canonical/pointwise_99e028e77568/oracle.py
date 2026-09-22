"""cuTile port of pointwise_99e028e77568: ShuffleNet dual BN + ReLU + channel-shuffle.

Two BN+ReLU branches feed a cat/channel-shuffle. Result is split back into
two outputs of shape [N, C, H, W] with cross-channel interleaving.

Structure: out[n, c, 0, hw] = relu(BN0(x0[n, c, h, w]))
           out[n, c, 1, hw] = relu(BN1(x1[n, c, h, w]))
Then out is viewed as [N, 2*C, H, W] and split into [N, C, H, W] + [N, C, H, W].

C=232 is not power-of-2; HW=49 also non-pow2. We tile per (n, c) with
BLOCK_HW = next_pow2(HW).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _shufflenet_dual_bn_relu_kernel(
    mean0_ptr,     # bf16 [C]
    x0_ptr,        # bf16 [N, C, HW]
    var0_ptr,      # bf16 [C]
    weight0_ptr,   # bf16 [C]
    bias0_ptr,     # bf16 [C]
    mean1_ptr,     # bf16 [C]
    x1_ptr,        # bf16 [N, C, HW]
    var1_ptr,      # bf16 [C]
    weight1_ptr,   # bf16 [C]
    bias1_ptr,     # bf16 [C]
    out_ptr,       # bf16 [N, C, 2, BLOCK_HW]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    def bn_relu(x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr):
        mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
        var = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
        weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
        bias = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)
        inv_std = ct.rsqrt(var + EPS)
        x_bf16 = ct.load(
            x_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_f = ct.astype(x_bf16, ct.float32)
        normalized = (x_f - mean) * inv_std
        affine = normalized * weight + bias
        affine_bf16 = ct.astype(affine, ct.bfloat16)
        # NaN-preserving ReLU
        is_nan = affine_bf16 != affine_bf16
        zero_bf16 = ct.astype(
            ct.full((1, 1, BLOCK_HW), 0.0, dtype=ct.float32), ct.bfloat16)
        relu_bf16 = ct.where(
            is_nan, affine_bf16,
            ct.where(affine_bf16 > zero_bf16, affine_bf16, zero_bf16))
        return relu_bf16

    r0 = bn_relu(x0_ptr, mean0_ptr, var0_ptr, weight0_ptr, bias0_ptr)
    r1 = bn_relu(x1_ptr, mean1_ptr, var1_ptr, weight1_ptr, bias1_ptr)

    ct.store(out_ptr, index=(n, c, 0, 0), tile=ct.reshape(r0, (1, 1, 1, BLOCK_HW)))
    ct.store(out_ptr, index=(n, c, 1, 0), tile=ct.reshape(r1, (1, 1, 1, BLOCK_HW)))


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@oracle_impl(hardware="B200", point="5e3665ee")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        arg5_1, arg6_1, arg7_1, arg8_1, arg9_1,
        _shape_param_0, _shape_param_1,
    ) = inputs
    N = int(arg1_1.shape[0])
    C = int(arg1_1.shape[1])
    H = int(arg1_1.shape[2])
    W = int(arg1_1.shape[3])
    HW = H * W
    BLOCK_HW = _next_pow2(HW)

    x0_flat = arg1_1.reshape(N, C, HW)
    x1_flat = arg6_1.reshape(N, C, HW)

    if BLOCK_HW == HW:
        out_padded = torch.empty(
            (N, C, 2, HW), device=arg1_1.device, dtype=torch.bfloat16)
    else:
        out_padded = torch.empty(
            (N, C, 2, BLOCK_HW), device=arg1_1.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C, 1),
        _shufflenet_dual_bn_relu_kernel,
        (
            arg0_1, x0_flat, arg2_1, arg3_1, arg4_1,
            arg5_1, x1_flat, arg7_1, arg8_1, arg9_1,
            out_padded, C, HW, BLOCK_HW,
        ),
    )

    if BLOCK_HW == HW:
        out = out_padded.view(N, C * 2, H, W).contiguous()
    else:
        out = out_padded[..., :HW].contiguous().view(N, C * 2, H, W)
    # Split [N, 2C, H, W] into two [N, C, H, W]
    return out[:, :C, :, :], out[:, C:, :, :]
