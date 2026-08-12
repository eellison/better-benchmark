"""cuTile port of pointwise_ef561a2def29: BN + ReLU + 3x3 stride-2 maxpool.

Reference: bf16 BN inference + NaN-preserving ReLU + 3x3 stride-2 pad-1
maxpool. HW=12544 is not power-of-2 so we run BN+ReLU in cuTile using a per
(n, c) tile of shape (1, 1, next_p2(HW)) with padded scratch, narrow back to
the valid region, and delegate the maxpool to torch (matches eager numerics
including NaN propagation).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HW_INPUT = 112 * 112  # 12544
BLOCK_HW = 16384


@ct.kernel
def _bn_relu_kernel(
    x_ptr,           # bf16 [N, C, HW]
    mean_ptr,        # bf16 [C]
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,         # bf16 [N, C, BLOCK_HW] scratch
    HW: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(
        x_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)

    mean_bc = ct.reshape(mean, (1, 1, 1))
    var_bc = ct.reshape(var, (1, 1, 1))
    weight_bc = ct.reshape(weight, (1, 1, 1))
    bias_bc = ct.reshape(bias, (1, 1, 1))

    invstd = 1.0 / ct.sqrt(var_bc + 1.0e-5)
    x_f = ct.astype(x, ct.float32)
    affine = (x_f - mean_bc) * invstd * weight_bc + bias_bc
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    affine_f32 = ct.astype(affine_bf16, ct.float32)
    zeros = ct.zeros(shape=(1, 1, BLOCK_HW_), dtype=ct.float32)
    is_nan = affine_f32 != affine_f32
    positive = affine_f32 > zeros
    relu_f = ct.where(is_nan | positive, affine_f32, zeros)
    relu_bf16 = ct.astype(relu_f, ct.bfloat16)
    ct.store(out_ptr, index=(n, c, 0), tile=relu_bf16)


@oracle_impl(hardware="B200", point="e4de5f8d", BLOCK_C=4, BLOCK_O=64)
@oracle_impl(hardware="B200", point="4194c732", BLOCK_C=4, BLOCK_O=64)
@oracle_impl(hardware="B200", point="9bacc993", BLOCK_C=4, BLOCK_O=64)
@oracle_impl(hardware="B200", point="aff40914", BLOCK_C=4, BLOCK_O=64)
def oracle_forward(inputs, **_kwargs):
    mean, x, var, weight, bias = inputs[:5]
    device = x.device
    N = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    HW = H * W

    out_pad = torch.empty((N, C, BLOCK_HW), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, C, 1), _bn_relu_kernel,
        (x.reshape(N, C, HW), mean, var, weight, bias, out_pad, HW, BLOCK_HW),
    )

    relu = out_pad.narrow(2, 0, HW).contiguous().view(N, C, H, W)
    out = torch.nn.functional.max_pool2d(relu, kernel_size=3, stride=2, padding=1)
    out_h = out.shape[2]
    out_w = out.shape[3]
    out_strided = torch.empty_strided(
        (N, C, out_h, out_w),
        (C * out_h * out_w, out_h * out_w, out_w, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_strided.copy_(out)
    return out_strided
