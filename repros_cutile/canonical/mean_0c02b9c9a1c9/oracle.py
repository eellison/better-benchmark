"""cuTile port of mean_0c02b9c9a1c9: DenseNet cat + BN + ReLU + spatial-mean.

Ports the Triton `_cat_bn_relu_mean_kernel`. We concatenate head+tail via torch
first (cheap materialization), then run a simple per-(batch, channel) reduction
that computes the spatial mean of ReLU(BN(x)) with an explicit bf16 rounding
before ReLU.

Batch=128, channels=184 (non-pow2), spatial=4x4 (HW=16). Grid iterates
(batch, channel), each program reduces 16 elements.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
HW = 16  # 4 * 4


@ct.kernel
def _cat_bn_relu_mean_kernel(
    x_ptr,       # bf16 [BATCH, CHANNELS, HW]
    mean_ptr,    # bf16 [CHANNELS]
    var_ptr,     # bf16 [CHANNELS]
    weight_ptr,  # bf16 [CHANNELS]
    bias_ptr,    # bf16 [CHANNELS]
    out_ptr,     # bf16 [BATCH, CHANNELS]
    HW_C: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)
    inv_std = ct.rsqrt(var + EPS)

    x = ct.astype(ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, HW_C)), ct.float32)
    y = (x - mean) * inv_std * weight + bias
    y_bf16 = ct.astype(y, ct.bfloat16)
    y_f32 = ct.astype(y_bf16, ct.float32)
    zero = ct.full((1, 1, HW_C), 0.0, dtype=ct.float32)
    relu = ct.where(y_f32 < 0.0, zero, y_f32)
    reduced = ct.sum(relu) * (1.0 / HW_C)
    ct.store(out_ptr, index=(n, c), tile=ct.astype(ct.reshape(reduced, (1, 1)), ct.bfloat16))


@oracle_impl(hardware="B200", point="2c3fa82b")
def oracle_forward(inputs):
    head, tail, mean, var, weight, bias, shape = inputs
    BATCH = 128
    CHANNELS = 184
    # Concat head+tail along channel dim: [128, 16+168, 4, 4]
    x = torch.cat([head, tail], dim=1)
    x_flat = x.view(BATCH, CHANNELS, HW)

    out = torch.empty_strided(
        (int(shape[0]), int(shape[1])),
        (int(shape[1]), 1),
        device=head.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, CHANNELS, 1),
        _cat_bn_relu_mean_kernel,
        (x_flat, mean, var, weight, bias, out, HW),
    )
    return out
