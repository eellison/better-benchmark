"""cuTile port of pointwise_a22d76c884fc: DenseNet 7-source cat + BN + ReLU.

We use torch.cat to materialize the channel concat outside the kernel,
then run a cuTile BN+ReLU kernel on the concatenated bf16 tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 (batch, C_OUT, H, W)
    mean_ptr,     # bf16 (C_OUT,)
    var_ptr,      # bf16 (C_OUT,)
    weight_ptr,   # bf16 (C_OUT,)
    bias_ptr,     # bf16 (C_OUT,)
    out_ptr,      # bf16 (batch, C_OUT, H, W)
    HW_PAD: ct.Constant[int],
    HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    var = ct.load(var_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    invstd = 1.0 / ct.sqrt(var_f + 1.0e-5)
    mean_3d = ct.reshape(mean_f, (1, 1, 1))
    invstd_3d = ct.reshape(invstd, (1, 1, 1))
    weight_3d = ct.reshape(weight_f, (1, 1, 1))
    bias_3d = ct.reshape(bias_f, (1, 1, 1))
    y = (x_f - mean_3d) * invstd_3d
    y = y * weight_3d + bias_3d
    y_bf16 = ct.astype(y, ct.bfloat16)
    zero_bf = ct.astype(ct.zeros((1, 1, HW_PAD), dtype=ct.float32), ct.bfloat16)
    relu = ct.where(y_bf16 < zero_bf, zero_bf, y_bf16)
    ct.store(out_ptr, index=(n, c, 0), tile=relu)


@oracle_impl(hardware="B200", point="60be6835", C0=512, C_OUT=704, H=7, W=7)
@oracle_impl(hardware="B200", point="6447e2c6", C0=256, C_OUT=448, H=14, W=14)
@oracle_impl(hardware="B200", point="20eb3168", C0=128, C_OUT=320, H=28, W=28)
@oracle_impl(hardware="B200", point="9869bb14", C0=64, C_OUT=256, H=56, W=56)
def oracle_forward(inputs, *, C0, C_OUT, H, W):
    x0, x1, x2, x3, x4, x5, x6, mean, var, weight, bias = inputs
    batch = int(x0.shape[0])
    # Materialize the 7-source cat via torch.
    cat = torch.cat([x0, x1, x2, x3, x4, x5, x6], dim=1)
    # Contiguous (batch, C_OUT, H, W)
    assert cat.shape[1] == C_OUT

    out = torch.empty_strided(
        (batch, C_OUT, H, W),
        (C_OUT * H * W, H * W, W, 1),
        device=cat.device,
        dtype=torch.bfloat16,
    )
    cat_3d = cat.view(batch, C_OUT, H * W)
    out_3d = out.view(batch, C_OUT, H * W)

    hw = H * W
    hw_pad = 1 << (hw - 1).bit_length()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, C_OUT, 1),
        _bn_relu_kernel,
        (cat_3d, mean, var, weight, bias, out_3d, hw_pad, hw),
    )
    return out
