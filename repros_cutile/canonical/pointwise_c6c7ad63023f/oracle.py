"""cuTile port of pointwise_c6c7ad63023f: DenseNet channel-cat + BN + NaN-preserving ReLU.

Uses torch to compute the channel cat, then a cuTile kernel to apply BN affine
(with bf16 boundary cast) + NaN-preserving ReLU per (batch, channel).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_relu_kernel(
    x_ptr,           # bf16 [rows, HW]
    mean_ptr,        # bf16 [C]
    var_ptr,         # bf16 [C]
    weight_ptr,      # bf16 [C]
    bias_ptr,        # bf16 [C]
    out_ptr,         # bf16 [rows, HW]
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    C: ct.Constant[int],
):
    row = ct.bid(0)
    channel = row - (row // C) * C

    x = ct.load(
        x_ptr, index=(row, 0), shape=(1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    x_f = ct.astype(x, ct.float32)
    invstd = 1.0 / ct.sqrt(var + EPS)
    y = (x_f - mean) * invstd * weight + bias
    # bf16 boundary cast then NaN-preserving ReLU on the bf16-rounded result.
    y_bf = ct.astype(y, ct.bfloat16)
    y_bf_f32 = ct.astype(y_bf, ct.float32)
    zero_f = ct.full((1, HW_PAD), 0.0, dtype=ct.float32)
    is_nan = y_bf_f32 != y_bf_f32
    max_val = ct.where(y_bf_f32 > zero_f, y_bf_f32, zero_f)
    relu = ct.where(is_nan, y_bf_f32, max_val)

    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid = cols < HW
    valid_2d = ct.reshape(valid, (1, HW_PAD))
    zero_bf = ct.full((1, HW_PAD), 0.0, dtype=ct.bfloat16)
    out_masked = ct.where(valid_2d, ct.astype(relu, ct.bfloat16), zero_bf)
    ct.store(out_ptr, index=(row, 0), tile=out_masked)


@oracle_impl(hardware="B200", point="7ea3cc48", C0=512, COUT=608, HW=49)
@oracle_impl(hardware="B200", point="6f284162", C0=256, COUT=352, HW=196)
@oracle_impl(hardware="B200", point="9cd6bc43", C0=128, COUT=224, HW=784)
@oracle_impl(hardware="B200", point="ea04eb28", C0=64, COUT=160, HW=3136)
def oracle_forward(inputs, *, C0: int, COUT: int, HW: int):
    x0, x1, x2, x3, mean, var, weight, bias = inputs
    batch = 64
    device = x0.device

    # Cat along channel dim via torch.
    # x0: [64, C0, ...], x1..x3: [64, 32, ...]
    # Reshape: viewing all as (batch, C, HW).
    # x0.shape possible [64, C0, H, W]. Reshape to 3D.
    def _to_flat(t):
        return t.contiguous().view(batch, -1, HW)

    x0_flat = _to_flat(x0)
    x1_flat = _to_flat(x1)
    x2_flat = _to_flat(x2)
    x3_flat = _to_flat(x3)
    cat = torch.cat([x0_flat, x1_flat, x2_flat, x3_flat], dim=1).contiguous()
    # cat shape [batch, COUT, HW]
    rows = batch * COUT
    x_2d = cat.view(rows, HW)

    if HW == 49:
        out = torch.empty_strided((64, COUT, 49), (COUT * 49, 49, 1),
                                  device=device, dtype=torch.bfloat16)
    else:
        out = torch.empty_strided((64, COUT, HW), (COUT * HW, HW, 1),
                                  device=device, dtype=torch.bfloat16)
    out_2d = out.view(rows, HW)

    HW_PAD = _next_pow2(HW)
    if HW == HW_PAD:
        stream = torch.cuda.current_stream()
        ct.launch(
            stream, (rows, 1, 1), _bn_relu_kernel,
            (x_2d, mean, var, weight, bias, out_2d,
             HW, HW_PAD, COUT),
        )
    else:
        out_buf = torch.empty((rows, HW), device=device, dtype=torch.bfloat16)
        stream = torch.cuda.current_stream()
        ct.launch(
            stream, (rows, 1, 1), _bn_relu_kernel,
            (x_2d, mean, var, weight, bias, out_buf,
             HW, HW_PAD, COUT),
        )
        out_2d.copy_(out_buf)

    # Reshape based on HW.
    if HW == 49:
        return out.view(64, COUT, 7, 7)
    if HW == 196:
        return out.view(64, COUT, 14, 14)
    if HW == 784:
        return out.view(64, COUT, 28, 28)
    return out.view(64, COUT, 56, 56)
