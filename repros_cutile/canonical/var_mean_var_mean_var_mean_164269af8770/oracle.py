"""cuTile port of var_mean_var_mean_var_mean_164269af8770: 6-branch Inception BN train + cat + avgpool.

Reference: 6 sibling training-BN branches (channels-last bf16 -> f32) with
population var_mean, affine+bf16+ReLU, cat into [128,2048,8,8], padded 3x3
avgpool tail, 12 running-stat copy_ updates.

Return: (mean0, rsqrt0, mean1, rsqrt1, mean2, rsqrt2, mean3, rsqrt3,
         mean4, rsqrt4, mean5, rsqrt5, cat, avg, 12 copy_ aliases).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
H = 8
W = 8
EPS = 0.001
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _avgpool_kernel(
    cat_ptr,      # bf16 [N*H*W, TOTAL_C_BC]
    avg_ptr,      # bf16 [N*H*W, TOTAL_C_BC]
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    TOTAL_C: ct.Constant[int],
    TOTAL_C_BC: ct.Constant[int],
):
    n = ct.bid(0)
    hw = ct.bid(1)
    out_h = hw // W_
    out_w = hw - out_h * W_

    acc = ct.zeros((TOTAL_C_BC,), dtype=ct.float32)
    for kh in ct.static_iter(range(3)):
        for kw in ct.static_iter(range(3)):
            in_h = out_h + kh - 1
            in_w = out_w + kw - 1
            valid = (in_h >= 0) & (in_h < H_) & (in_w >= 0) & (in_w < W_)
            safe_h = ct.where(valid, in_h, 0)
            safe_w = ct.where(valid, in_w, 0)
            row = n * H_ * W_ + safe_h * W_ + safe_w
            x = ct.load(cat_ptr, index=(row, 0), shape=(1, TOTAL_C_BC),
                         padding_mode=ct.PaddingMode.ZERO)
            xf = ct.astype(ct.reshape(x, (TOTAL_C_BC,)), ct.float32)
            xf = ct.where(valid, xf, 0.0)
            acc = acc + xf
    acc = acc * (1.0 / 9.0)

    cols = ct.arange(TOTAL_C_BC, dtype=ct.int32)
    active = cols < TOTAL_C
    acc_masked = ct.where(active, acc, 0.0)
    out_row = n * H_ * W_ + hw
    ct.store(avg_ptr, index=(out_row, 0),
              tile=ct.reshape(ct.astype(acc_masked, ct.bfloat16), (1, TOTAL_C_BC)))


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 [N*H*W, C_BC]
    mean_ptr,     # f32 [C_BC]
    invstd_ptr,   # f32 [C_BC]
    weight_ptr,   # f32 [C_BC]
    bias_ptr,     # f32 [C_BC]
    out_ptr,      # bf16 [N*H*W, C_BC]
    C: ct.Constant[int],
    C_BC: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, C_BC),
                 padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(ct.reshape(x, (C_BC,)), ct.float32)
    mean = ct.load(mean_ptr, index=(0,), shape=(C_BC,),
                    padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(C_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(C_BC,),
                      padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(C_BC,),
                    padding_mode=ct.PaddingMode.ZERO)
    y = (xf - mean) * invstd * weight + bias
    y_bf16_f = ct.astype(ct.astype(y, ct.bfloat16), ct.float32)
    is_nan = y_bf16_f != y_bf16_f
    relu = ct.where(is_nan, y_bf16_f,
                     ct.where(y_bf16_f > 0.0, y_bf16_f, 0.0))
    cols = ct.arange(C_BC, dtype=ct.int32)
    active = cols < C
    relu_masked = ct.where(active, relu, 0.0)
    ct.store(out_ptr, index=(row, 0),
              tile=ct.reshape(ct.astype(relu_masked, ct.bfloat16), (1, C_BC)))


@oracle_impl(hardware="B200", point="13e49e96")
def oracle_forward(inputs):
    (
        x0, rm0, rv0, w0, b0,
        x1, rm1, rv1, w1, b1,
        x2, rm2, rv2, w2, b2,
        x3, rm3, rv3, w3, b3,
        x4, rm4, rv4, w4, b4,
        x5, rm5, rv5, w5, b5,
    ) = inputs

    device = x0.device
    branches = [
        (x0, rm0, rv0, w0, b0),
        (x1, rm1, rv1, w1, b1),
        (x2, rm2, rv2, w2, b2),
        (x3, rm3, rv3, w3, b3),
        (x4, rm4, rv4, w4, b4),
        (x5, rm5, rv5, w5, b5),
    ]
    channels = [int(x.shape[1]) for x, *_ in branches]
    total_c = sum(channels)  # 2048

    means_1d = []
    invstds_1d = []
    means_4d = []
    rsqrts_4d = []
    y_bf16_list = []

    for (x, rm, rv, w, b), c in zip(branches, channels):
        xf = x.to(torch.float32)
        var, mean = torch.var_mean(xf, dim=[0, 2, 3], correction=0, keepdim=False)
        invstd = torch.rsqrt(var + EPS)

        # Running-stat update (mutable aliases).
        new_mean = rm * (1.0 - MOMENTUM) + mean * MOMENTUM
        new_var = rv * (1.0 - MOMENTUM) + var * RUNNING_VAR_CORRECTION * MOMENTUM
        torch.ops.aten.copy_(rm, new_mean)
        torch.ops.aten.copy_(rv, new_var)

        # Compose the [1,C,1,1] outputs.
        mean_4d = mean.view(1, c, 1, 1).clone()
        rsqrt_4d = invstd.view(1, c, 1, 1).clone()
        means_1d.append(mean)
        invstds_1d.append(invstd)
        means_4d.append(mean_4d)
        rsqrts_4d.append(rsqrt_4d)

    # Prepare padded channels-last-flat inputs.
    n = BATCH
    stream = torch.cuda.current_stream()
    for (x, rm, rv, w, b), c in zip(branches, channels):
        c_bc = _next_pow2(c)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(n * H * W, c)
        x_padded = torch.zeros(n * H * W, c_bc, device=device, dtype=torch.bfloat16)
        x_padded[:, :c] = x_flat
        mean_p = torch.zeros(c_bc, device=device, dtype=torch.float32)
        invstd_p = torch.zeros(c_bc, device=device, dtype=torch.float32)
        weight_p = torch.zeros(c_bc, device=device, dtype=torch.float32)
        bias_p = torch.zeros(c_bc, device=device, dtype=torch.float32)
        # find index of this branch
        idx = len(y_bf16_list)
        mean_p[:c] = means_1d[idx]
        invstd_p[:c] = invstds_1d[idx]
        weight_p[:c] = w.to(torch.float32)
        bias_p[:c] = b.to(torch.float32)

        y_flat = torch.empty(n * H * W, c_bc, device=device, dtype=torch.bfloat16)
        ct.launch(
            stream, (n * H * W, 1, 1),
            _bn_relu_kernel,
            (x_padded, mean_p, invstd_p, weight_p, bias_p, y_flat, c, c_bc),
        )
        y_bf16_list.append(y_flat[:, :c])

    # Build the cat: cat[N,total_c,H,W] channels-last.
    cat_shape = (n, total_c, H, W)
    cat_stride = (total_c * H * W, 1, W * total_c, total_c)
    cat = torch.empty_strided(cat_shape, cat_stride, device=device, dtype=torch.bfloat16)
    cat_flat = cat.permute(0, 2, 3, 1).view(n * H * W, total_c)
    off = 0
    for y_flat, c in zip(y_bf16_list, channels):
        cat_flat[:, off:off + c] = y_flat
        off += c

    # Avgpool via cuTile.
    total_c_bc = _next_pow2(total_c)
    cat_padded = torch.zeros(n * H * W, total_c_bc, device=device, dtype=torch.bfloat16)
    cat_padded[:, :total_c] = cat_flat
    avg_flat = torch.empty(n * H * W, total_c_bc, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (n, H * W, 1),
        _avgpool_kernel,
        (cat_padded, avg_flat, H, W, total_c, total_c_bc),
    )
    avg = torch.empty_strided(cat_shape, cat_stride, device=device, dtype=torch.bfloat16)
    avg.permute(0, 2, 3, 1).view(n * H * W, total_c).copy_(avg_flat[:, :total_c])

    return (
        means_4d[0], rsqrts_4d[0],
        means_4d[1], rsqrts_4d[1],
        means_4d[2], rsqrts_4d[2],
        means_4d[3], rsqrts_4d[3],
        means_4d[4], rsqrts_4d[4],
        means_4d[5], rsqrts_4d[5],
        cat, avg,
        rm0, rv0, rm1, rv1, rm2, rv2, rm3, rv3, rm4, rv4, rm5, rv5,
    )
