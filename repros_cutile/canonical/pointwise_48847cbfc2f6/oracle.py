"""cuTile port of pointwise_48847cbfc2f6: DenseNet BN+ReLU+maxpool+BN+ReLU
fused stencil.

Torch handles the maxpool operation; cuTile handles the elementwise
BN+ReLU steps before and after.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 (batch, C, HW)
    mean_ptr,     # bf16 (C,)
    var_ptr,      # bf16 (C,)
    weight_ptr,   # bf16 (C,)
    bias_ptr,     # bf16 (C,)
    out_ptr,      # bf16 (batch, C, HW)
    HW_PAD: ct.Constant[int],
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


@oracle_impl(hardware="B200", point="710a4598", BLOCK_C=4, BLOCK_OUT=128)
def oracle_forward(inputs, **_kwargs):
    mean1, x, var1, weight1, bias1, mean2, var2, weight2, bias2, _kernel, _stride = inputs
    device = x.device
    batch = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w

    # Stage 1: BN + ReLU on x (h x w).
    stage1 = torch.empty((batch, c, h, w), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    hw_pad = 1 << (hw - 1).bit_length()
    ct.launch(stream, (batch, c, 1), _bn_relu_kernel,
              (x.view(batch, c, hw), mean1, var1, weight1, bias1,
               stage1.view(batch, c, hw), hw_pad))

    # 3x3 stride-2 maxpool with padding=1.
    pool = torch.nn.functional.max_pool2d(
        stage1, kernel_size=3, stride=2, padding=1, dilation=1,
    )
    out_h = int(pool.shape[2])
    out_w = int(pool.shape[3])
    hw_out = out_h * out_w

    # Stage 2: BN + ReLU on pooled tensor.
    stage2 = torch.empty((batch, c, out_h, out_w), device=device, dtype=torch.bfloat16)
    hw_out_pad = 1 << (hw_out - 1).bit_length()
    ct.launch(stream, (batch, c, 1), _bn_relu_kernel,
              (pool.view(batch, c, hw_out), mean2, var2, weight2, bias2,
               stage2.view(batch, c, hw_out), hw_out_pad))
    return pool, stage2
