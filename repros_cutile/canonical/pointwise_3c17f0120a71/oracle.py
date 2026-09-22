"""cuTile port of pointwise_3c17f0120a71: ShuffleNet dual BN + ReLU + interleave shuffle.

For each (n, c, h, w): compute BN+ReLU on x_a and x_b, then interleave-store
  out[n, 2*c,   h, w] = a
  out[n, 2*c+1, h, w] = b
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
C = 116
H = 14
W = 14
HP = 16
WP = 16
HW = H * W
OUT_C = 2 * C  # 232


@ct.kernel
def _dual_bn_relu_shuffle_kernel(
    mean_a_ptr,
    x_a_ptr,
    var_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    mean_b_ptr,
    x_b_ptr,
    var_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    out_ptr,   # [BATCH, OUT_C, H, W]
    H: ct.Constant[int],
    W: ct.Constant[int],
    HP: ct.Constant[int],
    WP: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    x_a = ct.load(x_a_ptr, index=(n, c, 0, 0), shape=(1, 1, HP, WP),
                  padding_mode=ct.PaddingMode.ZERO)
    x_b = ct.load(x_b_ptr, index=(n, c, 0, 0), shape=(1, 1, HP, WP),
                  padding_mode=ct.PaddingMode.ZERO)
    x_a = ct.astype(ct.reshape(x_a, (HP, WP)), ct.float32)
    x_b = ct.astype(ct.reshape(x_b, (HP, WP)), ct.float32)

    mean_a = ct.astype(ct.load(mean_a_ptr, index=(c,), shape=(1,)), ct.float32)
    var_a = ct.astype(ct.load(var_a_ptr, index=(c,), shape=(1,)), ct.float32)
    wa = ct.astype(ct.load(weight_a_ptr, index=(c,), shape=(1,)), ct.float32)
    ba = ct.astype(ct.load(bias_a_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_b = ct.astype(ct.load(mean_b_ptr, index=(c,), shape=(1,)), ct.float32)
    var_b = ct.astype(ct.load(var_b_ptr, index=(c,), shape=(1,)), ct.float32)
    wb = ct.astype(ct.load(weight_b_ptr, index=(c,), shape=(1,)), ct.float32)
    bb = ct.astype(ct.load(bias_b_ptr, index=(c,), shape=(1,)), ct.float32)

    inv_a = ct.rsqrt(var_a + 1.0e-5)
    inv_b = ct.rsqrt(var_b + 1.0e-5)
    ya = (x_a - mean_a) * inv_a * wa + ba
    yb = (x_b - mean_b) * inv_b * wb + bb
    ya_bf = ct.astype(ya, ct.bfloat16)
    yb_bf = ct.astype(yb, ct.bfloat16)
    zero_bf = ct.full(shape=(HP, WP), fill_value=0.0, dtype=ct.bfloat16)
    a = ct.where(ya_bf < zero_bf, zero_bf, ya_bf)
    b = ct.where(yb_bf < zero_bf, zero_bf, yb_bf)

    # Store back only H x W via scatter since HP/WP > H/W.
    h_idx = ct.arange(HP, dtype=ct.int32)
    w_idx = ct.arange(WP, dtype=ct.int32)
    h_valid = ct.reshape(h_idx < H, (HP, 1))
    w_valid = ct.reshape(w_idx < W, (1, WP))
    valid = h_valid & w_valid

    # 4D scatter
    n_i = ct.full(shape=(HP, WP), fill_value=n, dtype=ct.int32)
    c_a = ct.full(shape=(HP, WP), fill_value=2 * c, dtype=ct.int32)
    c_b = ct.full(shape=(HP, WP), fill_value=2 * c + 1, dtype=ct.int32)
    hh = ct.reshape(h_idx, (HP, 1))
    ww = ct.reshape(w_idx, (1, WP))
    hh_bcast = ct.full(shape=(HP, WP), fill_value=0, dtype=ct.int32) + hh
    ww_bcast = ct.full(shape=(HP, WP), fill_value=0, dtype=ct.int32) + ww
    ct.scatter(out_ptr, (n_i, c_a, hh_bcast, ww_bcast), a, mask=valid)
    ct.scatter(out_ptr, (n_i, c_b, hh_bcast, ww_bcast), b, mask=valid)


@oracle_impl(hardware="B200", point="f47c0655", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    mean_a, x_a, var_a, weight_a, bias_a, mean_b, x_b, var_b, weight_b, bias_b, _shape0, _shape1 = inputs
    shuffled = torch.empty_strided(
        (BATCH, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=x_a.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, C, 1),
        _dual_bn_relu_shuffle_kernel,
        (mean_a, x_a, var_a, weight_a, bias_a,
         mean_b, x_b, var_b, weight_b, bias_b,
         shuffled, H, W, HP, WP),
    )
    return shuffled[:, :C, :, :], shuffled[:, C:, :, :]
