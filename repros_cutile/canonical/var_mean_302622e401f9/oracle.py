"""cuTile port of var_mean_302622e401f9: Adv-Inception BN train + maxpool + avgpool.

Reference:
  var_mean(x.float(), [0,2,3], correction=0) over channels-last [128,192,71,71]
  invstd = rsqrt(var + 0.001)
  affine = ((x - mean) * invstd * weight + bias).to(bf16); relu preserving NaN
  maxpool 3x3 stride=2 -> [128,192,35,35] channels-last, plus int8 offsets
  avgpool 3x3 stride=1 pad=1 on maxpool -> same shape channels-last
  return (mean, invstd, pool_values, pool_offsets, avg, running_mean, running_var).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 192
H_IN = 71
W_IN = 71
H_OUT = 35
W_OUT = 35
EPS = 0.001
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.00000154979411
# Channels 192 -> pad to 256 (next pow2)
BLOCK_C = 256


@ct.kernel
def _bn_relu_maxpool_kernel(
    x_ptr,          # bf16 (128, 71, 71, 192) memory in channels-last
    mean_ptr,       # f32 [192]
    invstd_ptr,     # f32 [192]
    weight_ptr,     # f32 [192]
    bias_ptr,       # f32 [192]
    pool_ptr,       # bf16 (128, 35, 35, 192) memory (physical NHWC)
    offsets_ptr,    # i8 (128, 35, 35, 192)
    C: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    # Grid (batch, out_hw, 1) where out_hw = 35*35 = 1225
    batch = ct.bid(0)
    s = ct.bid(1)

    # Load per-channel affine params, padded to BLOCK_C_
    cols = ct.arange(BLOCK_C_, dtype=ct.int32)
    active_c = cols < C
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C_,),
                    padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C_,),
                      padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C_,),
                      padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C_,),
                    padding_mode=ct.PaddingMode.ZERO)

    # Recover out_h, out_w
    out_h = s // 35
    out_w = s - out_h * 35

    best = ct.full((BLOCK_C_,), -3.4e38, dtype=ct.float32)
    best_idx = ct.zeros((BLOCK_C_,), dtype=ct.int32)

    for kh in ct.static_iter(range(3)):
        for kw in ct.static_iter(range(3)):
            in_h = out_h * 2 + kh
            in_w = out_w * 2 + kw
            # tensor x is (N, H_IN, W_IN, C) physically; load a (1,BLOCK_C_) tile
            row_idx = batch * H_IN * W_IN + in_h * W_IN + in_w
            x = ct.load(x_ptr, index=(row_idx, 0), shape=(1, BLOCK_C_),
                         padding_mode=ct.PaddingMode.ZERO)
            xf = ct.astype(ct.reshape(x, (BLOCK_C_,)), ct.float32)
            centered = xf - mean
            normalized = centered * invstd
            affine = normalized * weight + bias
            rounded_bf16 = ct.astype(affine, ct.bfloat16)
            rounded = ct.astype(rounded_bf16, ct.float32)
            # ReLU preserving NaN: if x != x -> nan; else max(x, 0)
            is_nan = rounded != rounded
            relu = ct.where(is_nan, rounded,
                             ct.where(rounded > 0.0, rounded, 0.0))
            take = (relu > best) | is_nan
            best = ct.where(take, relu, best)
            best_idx = ct.where(take, kh * 3 + kw, best_idx)

    # Mask off inactive channels: set to 0 for out-of-range channels
    best_masked = ct.where(active_c, best, 0.0)
    best_bf16 = ct.astype(best_masked, ct.bfloat16)
    idx_i8 = ct.astype(ct.where(active_c, best_idx, 0), ct.int8)

    out_row = batch * H_OUT * W_OUT + s
    ct.store(pool_ptr, index=(out_row, 0),
              tile=ct.reshape(best_bf16, (1, BLOCK_C_)))
    ct.store(offsets_ptr, index=(out_row, 0),
              tile=ct.reshape(idx_i8, (1, BLOCK_C_)))


@ct.kernel
def _avgpool_kernel(
    pool_ptr,     # bf16 (128, 35, 35, 192)
    avg_ptr,      # bf16 (128, 35, 35, 192)
    C: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    batch = ct.bid(0)
    s = ct.bid(1)

    cols = ct.arange(BLOCK_C_, dtype=ct.int32)
    active_c = cols < C

    out_h = s // W_OUT
    out_w = s - out_h * W_OUT

    acc = ct.zeros((BLOCK_C_,), dtype=ct.float32)
    for kh in ct.static_iter(range(3)):
        for kw in ct.static_iter(range(3)):
            in_h = out_h + kh - 1
            in_w = out_w + kw - 1
            valid = (in_h >= 0) & (in_h < H_OUT) & (in_w >= 0) & (in_w < W_OUT)
            # Load using bounded index
            safe_h = ct.where(valid, in_h, 0)
            safe_w = ct.where(valid, in_w, 0)
            row_idx = batch * H_OUT * W_OUT + safe_h * W_OUT + safe_w
            x = ct.load(pool_ptr, index=(row_idx, 0), shape=(1, BLOCK_C_),
                         padding_mode=ct.PaddingMode.ZERO)
            xf = ct.astype(ct.reshape(x, (BLOCK_C_,)), ct.float32)
            xf = ct.where(valid, xf, 0.0)
            acc = acc + xf

    acc = acc * (1.0 / 9.0)
    acc_masked = ct.where(active_c, acc, 0.0)
    acc_bf16 = ct.astype(acc_masked, ct.bfloat16)
    out_row = batch * H_OUT * W_OUT + s
    ct.store(avg_ptr, index=(out_row, 0),
              tile=ct.reshape(acc_bf16, (1, BLOCK_C_)))


@oracle_impl(hardware="B200", point="9b65830c")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    device = arg0_1.device

    # Channels-last: arg0_1 shape [128,192,71,71] stride [967872, 1, 13632, 192]
    # Reshape as (N, H_IN*W_IN, C) contiguous by permute.
    # arg0_1 physical order is N,H,W,C (channels-last)
    x_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()  # (N,H,W,C)
    x_flat = x_nhwc.view(BATCH * H_IN * W_IN, CHANNELS)

    # Compute var_mean using torch (fp32 pop stats over [0,2,3]).
    x_f32 = arg0_1.to(torch.float32)
    var, mean = torch.var_mean(x_f32, dim=[0, 2, 3], correction=0, keepdim=False)
    invstd = torch.rsqrt(var + EPS)

    # Running stats
    new_mean = arg1_1 * (1.0 - MOMENTUM) + mean * MOMENTUM
    new_var = arg2_1 * (1.0 - MOMENTUM) + var * RUNNING_VAR_CORRECTION * MOMENTUM
    torch.ops.aten.copy_(arg1_1, new_mean)
    torch.ops.aten.copy_(arg2_1, new_var)

    # Prepare outputs (channels-last strided)
    mean_out = torch.empty_strided(
        (1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1),
        device=device, dtype=torch.float32,
    )
    mean_out.view(-1).copy_(mean)
    invstd_out = torch.empty_strided(
        (1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1),
        device=device, dtype=torch.float32,
    )
    invstd_out.view(-1).copy_(invstd)

    weight_f = arg3_1.to(torch.float32)
    bias_f = arg4_1.to(torch.float32)

    pool_shape = (BATCH, CHANNELS, H_OUT, W_OUT)
    pool_stride = (CHANNELS * H_OUT * W_OUT, 1, W_OUT * CHANNELS, CHANNELS)
    pool = torch.empty_strided(pool_shape, pool_stride, device=device, dtype=torch.bfloat16)
    pool_offsets = torch.empty_strided(pool_shape, pool_stride, device=device, dtype=torch.int8)
    avg = torch.empty_strided(pool_shape, pool_stride, device=device, dtype=torch.bfloat16)

    # Physical NHWC 2D views of the outputs (128*35*35 rows, 192 cols)
    pool_2d = pool.permute(0, 2, 3, 1).contiguous().view(BATCH * H_OUT * W_OUT, CHANNELS)
    offsets_2d = pool_offsets.permute(0, 2, 3, 1).contiguous().view(BATCH * H_OUT * W_OUT, CHANNELS)
    avg_2d = avg.permute(0, 2, 3, 1).contiguous().view(BATCH * H_OUT * W_OUT, CHANNELS)

    stream = torch.cuda.current_stream()
    out_hw = H_OUT * W_OUT

    ct.launch(
        stream,
        (BATCH, out_hw, 1),
        _bn_relu_maxpool_kernel,
        (
            x_flat, mean, invstd, weight_f, bias_f,
            pool_2d, offsets_2d, CHANNELS, BLOCK_C,
        ),
    )
    ct.launch(
        stream,
        (BATCH, out_hw, 1),
        _avgpool_kernel,
        (pool_2d, avg_2d, CHANNELS, BLOCK_C),
    )

    # Copy pool_2d back into channels-last strided pool.
    pool_2d_view = pool_2d.view(BATCH, H_OUT, W_OUT, CHANNELS).permute(0, 3, 1, 2)
    pool.copy_(pool_2d_view)
    offsets_view = offsets_2d.view(BATCH, H_OUT, W_OUT, CHANNELS).permute(0, 3, 1, 2)
    pool_offsets.copy_(offsets_view)
    avg_view = avg_2d.view(BATCH, H_OUT, W_OUT, CHANNELS).permute(0, 3, 1, 2)
    avg.copy_(avg_view)

    return mean_out, invstd_out, pool, pool_offsets, avg, arg1_1, arg2_1
