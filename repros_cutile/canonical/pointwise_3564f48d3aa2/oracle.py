"""cuTile port of pointwise_3564f48d3aa2: DenseNet 6-branch cat + BN-inference + ReLU.

Pre-concatenates the 6 branch inputs via torch.cat (contiguous NCHW), then
runs a single cuTile kernel for BN-inference (channel-broadcast affine),
bf16 rounding, and ReLU. The output preserves the eager NCHW contiguous
layout: (N, C_TOTAL, H, W).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    x_ptr,       # (N, C, HW) bf16 (contiguous NCHW view)
    mean_ptr,    # (C,) bf16
    var_ptr,     # (C,) bf16
    weight_ptr,  # (C,) bf16
    bias_ptr,    # (C,) bf16
    out_ptr,     # (N, C, HW) bf16 (contiguous)
    HW: ct.Constant[int],
    HW_PADDED: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    # Load HW spatial elements at (n, c). Pad up to power-of-2 tile with 0.
    x = ct.load(
        x_ptr, index=(n, c, 0), shape=(1, 1, HW_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    m = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    v = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    w = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    b = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)

    invstd = 1.0 / ct.sqrt(v + 1.0e-5)
    m3 = ct.reshape(m, (1, 1, 1))
    invstd3 = ct.reshape(invstd, (1, 1, 1))
    w3 = ct.reshape(w, (1, 1, 1))
    b3 = ct.reshape(b, (1, 1, 1))
    y = (x_f - m3) * invstd3 * w3 + b3
    y_bf = ct.astype(y, ct.bfloat16)
    # NaN-preserving ReLU: tl.where(y < 0, 0, y) → NaN < 0 is False → keeps NaN.
    y_bf_f = ct.astype(y_bf, ct.float32)
    y_relu_f = ct.where(y_bf_f < 0.0, 0.0, y_bf_f)
    y_relu = ct.astype(y_relu_f, ct.bfloat16)
    # Store: cuTile drops OOB stores for tail elements past HW.
    ct.store(out_ptr, index=(n, c, 0), tile=y_relu)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


@oracle_impl(hardware="B200", point="a0c86fd4")
@oracle_impl(hardware="B200", point="64fc1db2")
@oracle_impl(hardware="B200", point="5ababadb")
@oracle_impl(hardware="B200", point="994d5473")
def oracle_forward(inputs):
    x0, x1, x2, x3, x4, x5, mean, var, weight, bias = inputs
    # Pre-concatenate branches along channel dim → contiguous NCHW.
    cat = torch.cat([x0, x1, x2, x3, x4, x5], dim=1).contiguous()
    n, c_total, h, w = cat.shape
    hw = h * w
    hw_padded = _next_pow2(hw)
    # View as (N, C, HW) so the last axis is HW (contiguous).
    cat3 = cat.view(n, c_total, hw)
    out3 = torch.empty(
        (n, c_total, hw), device=x0.device, dtype=torch.bfloat16
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n, c_total, 1),
        _bn_relu_kernel,
        (cat3, mean, var, weight, bias, out3, hw, hw_padded),
    )
    # View back to (N, C, H, W).
    return out3.view(n, c_total, h, w)
