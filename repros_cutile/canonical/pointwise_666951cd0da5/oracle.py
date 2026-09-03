"""cuTile port of pointwise_666951cd0da5: ShuffleNetV2 BN + ReLU + channel-shuffle.

Runs the BN+ReLU pointwise producer as a cuTile kernel (per element), then does
the cat/reshape/permute/reshape/split channel-shuffle using torch layout ops.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 116
H = 14
W = 14
HW = H * W
NUMEL = 64 * C * HW  # 64 * 116 * 196
BLOCK = 1024  # divides 64*196 chunks; NUMEL % BLOCK not exact, so use padding-mode zero


@ct.kernel
def _bn_relu_kernel(
    conv_ptr,      # bf16 [N, C, HW]  (viewed as 3D)
    mean_ptr,      # bf16 [C]
    var_ptr,       # bf16 [C]
    weight_ptr,    # bf16 [C]
    bias_ptr,      # bf16 [C]
    out_ptr,       # bf16 [N, C, HW]
    BLOCK_C: ct.Constant[int],
):
    # Grid: (N, C, cdiv(HW, BLOCK_C))
    n = ct.bid(0)
    c = ct.bid(1)
    h_block = ct.bid(2)

    x_bf = ct.load(conv_ptr, index=(n, c, h_block), shape=(1, 1, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)

    mean_bf = ct.load(mean_ptr, index=(c,), shape=(1,))
    var_bf = ct.load(var_ptr, index=(c,), shape=(1,))
    weight_bf = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_bf = ct.load(bias_ptr, index=(c,), shape=(1,))

    mean = ct.astype(mean_bf, ct.float32)
    var = ct.astype(var_bf, ct.float32)
    weight = ct.astype(weight_bf, ct.float32)
    bias = ct.astype(bias_bf, ct.float32)

    centered = x - ct.reshape(mean, (1, 1, 1))
    invstd = ct.rsqrt(var + 1.0e-5)
    normalized = centered * ct.reshape(invstd, (1, 1, 1)) * ct.reshape(weight, (1, 1, 1)) + ct.reshape(bias, (1, 1, 1))
    normalized_bf = ct.astype(normalized, ct.bfloat16)
    # ReLU preserving NaN: where (x > 0) | (x != x), x, 0
    zero_bf = ct.zeros((1, 1, BLOCK_C), dtype=ct.bfloat16)
    y = ct.where((normalized_bf > zero_bf) | (normalized_bf != normalized_bf), normalized_bf, zero_bf)

    ct.store(out_ptr, index=(n, c, h_block), tile=y)


@oracle_impl(hardware="B200", point="1e6a9948")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape0, _shape1 = inputs
    # arg0=mean bf16[116], arg1=conv bf16[64,116,14,14], arg2=var bf16[116],
    # arg3=weight bf16[116], arg4=bias bf16[116], arg5=left bf16[64,116,14,14]
    n = int(arg1_1.shape[0])
    c = int(arg1_1.shape[1])
    h = int(arg1_1.shape[2])
    w = int(arg1_1.shape[3])
    hw = h * w
    device = arg1_1.device

    # Compute BN+ReLU on the conv branch via cuTile.
    bn_out = torch.empty((n, c, h, w), device=device, dtype=torch.bfloat16)
    # Use tile over the last dim. BLOCK_C must be pow2. HW=196; use 256 with padding.
    BLOCK_C = 256
    conv_3d = arg1_1.contiguous().view(n, c, hw)
    bn_out_3d = bn_out.view(n, c, hw)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n, c, ct.cdiv(hw, BLOCK_C)),
        _bn_relu_kernel,
        (conv_3d, arg0_1, arg2_1, arg3_1, arg4_1, bn_out_3d, BLOCK_C),
    )

    # Do the ShuffleNetV2 channel shuffle via torch.
    cat = torch.cat([arg5_1, bn_out], dim=1)  # (N, 2C, H, W)
    view = cat.view(n, 2, c, h, w)
    permuted = view.permute(0, 2, 1, 3, 4).contiguous()
    view_1 = permuted.view(n, 2 * c, h, w)
    getitem_0 = view_1[:, :c, :, :]
    getitem_1 = view_1[:, c:, :, :]
    return getitem_0, getitem_1
