"""cuTile port of pointwise_e4b7b0947416: Inception 4-branch BN-inference + ReLU + cat.

Each branch is a channels-last bf16 (N, Cb, H, W) tensor. Applies per-channel
BN-inference (mean/var/weight/bias in bf16, computed via fp32 sqrt+recip with
eps=0.001), bf16 rounding, then ReLU. The four bf16-rounded ReLU tensors are
concatenated along the channel axis into a channels-last (N, sum(Cb), H, W).

Strategy: run one cuTile kernel per branch, writing directly into a slice of
a pre-allocated channels-last output that combines all four branches. Non-pow2
H, W (17, 35) are handled with padded tiles + OOB stores dropped by cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


@ct.kernel
def _branch_bn_relu_kernel(
    x_ptr,       # (N, C, H, W) bf16
    mean_ptr,    # (C,) bf16
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,     # (N, C, H, W) bf16 (channels-last stride, slice of shared out)
    H_PADDED: ct.Constant[int],
    W_PADDED: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    x = ct.load(
        x_ptr, index=(n, c, 0, 0), shape=(1, 1, H_PADDED, W_PADDED),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    m = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    v = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    w = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    b = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)

    invstd = 1.0 / ct.sqrt(v + EPS)
    m4 = ct.reshape(m, (1, 1, 1, 1))
    invstd4 = ct.reshape(invstd, (1, 1, 1, 1))
    w4 = ct.reshape(w, (1, 1, 1, 1))
    b4 = ct.reshape(b, (1, 1, 1, 1))
    y = (x_f - m4) * invstd4 * w4 + b4
    y_bf = ct.astype(y, ct.bfloat16)
    # NaN-preserving ReLU: cvt f32 → where(y < 0, 0, y) → cvt bf16.
    y_bf_f = ct.astype(y_bf, ct.float32)
    y_relu_f = ct.where(y_bf_f < 0.0, 0.0, y_bf_f)
    y_relu = ct.astype(y_relu_f, ct.bfloat16)
    ct.store(out_ptr, index=(n, c, 0, 0), tile=y_relu)


def _run_branch(x, mean, var, weight, bias, out_slice):
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n, c, 1),
        _branch_bn_relu_kernel,
        (x, mean, var, weight, bias, out_slice, _next_pow2(h), _next_pow2(w)),
    )


@oracle_impl(hardware="B200", point="25fb017b")
@oracle_impl(hardware="B200", point="78045192")
def oracle_forward(inputs, **kwargs):
    (
        mean0, x0, var0, weight0, bias0,
        mean1, x1, var1, weight1, bias1,
        mean2, x2, var2, weight2, bias2,
        mean3, x3, var3, weight3, bias3,
    ) = inputs
    n = int(x0.shape[0])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    hw = h * w
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    c2 = int(x2.shape[1])
    c3 = int(x3.shape[1])
    c_total = c0 + c1 + c2 + c3

    out = torch.empty_strided(
        (n, c_total, h, w),
        (c_total * hw, 1, w * c_total, c_total),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    slice0 = out.narrow(1, 0, c0)
    slice1 = out.narrow(1, c0, c1)
    slice2 = out.narrow(1, c0 + c1, c2)
    slice3 = out.narrow(1, c0 + c1 + c2, c3)

    _run_branch(x0, mean0, var0, weight0, bias0, slice0)
    _run_branch(x1, mean1, var1, weight1, bias1, slice1)
    _run_branch(x2, mean2, var2, weight2, bias2, slice2)
    _run_branch(x3, mean3, var3, weight3, bias3, slice3)

    return out
