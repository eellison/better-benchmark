"""cuTile port of var_mean_mean_d205976abf2e: 3-kernel BN training + ReLU +
channel-cat + spatial mean (GhostNet).

Eager:
  x_f32 = x.to(f32)
  var, mean = var_mean(x_f32, [0, 2, 3], correction=0, keepdim=True)
  invstd = rsqrt(var + 1e-5)
  affine_bf16 = ((x - mean) * invstd * weight + bias).to(bf16)
  relu = relu(affine_bf16)
  cat = cat([residual, relu], dim=1)     # [N, 2*C, H, W]
  spatial_mean = mean(cat, [2, 3], keepdim=True)  # bf16 [N, 2*C, 1, 1]
  running_mean.copy_(mean * 0.1 + running_mean * 0.9)
  running_var.copy_(var * CORRECTION * 0.1 + running_var * 0.9)
  return (getitem_1_1x1x1, rsqrt_1x1x1, cat, spatial_mean, rmean, rvar)

Kernels:
  1) partial-stats over NHWC-flat (E, C) view
  2) BN-affine + ReLU + cat writer + spatial-mean into flat (N, 2*C, HW) output

We handle both channels-last and NCHW inputs by materializing NHWC-contig
copies of x and residual (this is what the Triton graph effectively expects).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
CORRECTION = 1.0000398612827361


@ct.kernel
def _partial_stats_kernel(
    x_2d_ptr,          # bf16 [E, C]
    partial_sum_ptr,   # f32 [NUM_CHUNKS, C]
    partial_sum2_ptr,  # f32 [NUM_CHUNKS, C]
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    chunk = ct.bid(0)
    ch_tile = ct.bid(1)
    x_bf = ct.load(
        x_2d_ptr,
        index=(chunk, ch_tile),
        shape=(BLOCK_R_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)
    s = ct.sum(x, axis=0)
    s2 = ct.sum(x * x, axis=0)
    ct.store(partial_sum_ptr, index=(chunk, ch_tile), tile=ct.reshape(s, (1, BLOCK_C_)))
    ct.store(partial_sum2_ptr, index=(chunk, ch_tile), tile=ct.reshape(s2, (1, BLOCK_C_)))


@ct.kernel
def _affine_relu_cat_mean_kernel(
    x_ptr,              # bf16 [N, HW, C]
    residual_ptr,       # bf16 [N, HW, C]
    mean_ptr,           # f32 [C]
    invstd_ptr,         # f32 [C]
    weight_ptr,         # f32 [C]
    bias_ptr,           # f32 [C]
    cat_ptr,            # bf16 [N, HW, 2*C]
    spatial_mean_ptr,   # bf16 [N, 2*C]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    C: ct.Constant[int],
):
    n = ct.bid(0)
    c_tile = ct.bid(1)

    # Load one (HW_BLOCK, C_BLOCK) tile of x and residual for this (batch, cblock).
    x_bf = ct.load(
        x_ptr,
        index=(n, 0, c_tile),
        shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual_bf = ct.load(
        residual_ptr,
        index=(n, 0, c_tile),
        shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Per-channel scalars, shape (1, 1, BLOCK_C).
    mean_1d = ct.load(mean_ptr, index=(c_tile,), shape=(BLOCK_C,))
    invstd_1d = ct.load(invstd_ptr, index=(c_tile,), shape=(BLOCK_C,))
    weight_1d = ct.load(weight_ptr, index=(c_tile,), shape=(BLOCK_C,))
    bias_1d = ct.load(bias_ptr, index=(c_tile,), shape=(BLOCK_C,))
    mean = ct.reshape(mean_1d, (1, 1, BLOCK_C))
    invstd = ct.reshape(invstd_1d, (1, 1, BLOCK_C))
    weight = ct.reshape(weight_1d, (1, 1, BLOCK_C))
    bias = ct.reshape(bias_1d, (1, 1, BLOCK_C))

    x = ct.astype(x_bf, ct.float32)
    affine = (x - mean) * invstd * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf, ct.float32)  # bf16 round-trip
    zero_f = ct.zeros((1, BLOCK_HW, BLOCK_C), dtype=ct.float32)
    relu_f = ct.where(affine_f > 0.0, affine_f, zero_f)

    # Masks for HW and channel validity.
    hw_range = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_valid = ct.reshape(hw_range < HW, (1, BLOCK_HW, 1))
    c_range = ct.arange(BLOCK_C, dtype=ct.int32) + c_tile * BLOCK_C
    c_valid = ct.reshape(c_range < C, (1, 1, BLOCK_C))
    tile_valid = hw_valid & c_valid

    relu_bf = ct.astype(ct.where(tile_valid, relu_f, zero_f), ct.bfloat16)
    # Cat writes: residual at channels [0, C) and relu at channels [C, 2*C).
    # cat storage is [N, HW, 2*C] contiguous with 2*C stride.
    # Write residual half.
    ct.store(cat_ptr, index=(n, 0, c_tile), tile=residual_bf)
    # Write relu half — index c_tile shifted by (C / BLOCK_C) tiles.
    ct.store(cat_ptr, index=(n, 0, c_tile + C // BLOCK_C), tile=relu_bf)

    # Spatial mean over HW: sum reduce along axis=1 then divide by HW.
    residual_f = ct.astype(residual_bf, ct.float32)
    residual_f_masked = ct.where(tile_valid, residual_f, zero_f)
    residual_sum = ct.sum(residual_f_masked, axis=1)  # (1, BLOCK_C)
    residual_mean = residual_sum * (1.0 / HW)

    relu_masked = ct.where(tile_valid, relu_f, zero_f)
    relu_sum = ct.sum(relu_masked, axis=1)
    relu_mean = relu_sum * (1.0 / HW)

    residual_mean_bf = ct.astype(residual_mean, ct.bfloat16)
    relu_mean_bf = ct.astype(relu_mean, ct.bfloat16)

    # Store into spatial_mean at (n, c_tile) and (n, c_tile + C/BLOCK_C).
    ct.store(spatial_mean_ptr, index=(n, c_tile), tile=residual_mean_bf)
    ct.store(spatial_mean_ptr, index=(n, c_tile + C // BLOCK_C), tile=relu_mean_bf)


BLOCK_STAT_R = 2048
BLOCK_STAT_C = 16


def _pow2_ceil(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _forward(inputs):
    x, running_mean, running_var, weight, bias, residual = inputs
    N = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    HW = H * W
    E = N * HW
    device = x.device

    # Materialize NHWC-contig views.
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    residual_nhwc = residual.permute(0, 2, 3, 1).contiguous()
    x_2d = x_nhwc.view(E, C)
    x_3d = x_nhwc.view(N, HW, C)
    residual_3d = residual_nhwc.view(N, HW, C)

    # Partial stats.
    num_chunks = (E + BLOCK_STAT_R - 1) // BLOCK_STAT_R
    partial_sum = torch.empty((num_chunks, C), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_chunks, C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, ct.cdiv(C, BLOCK_STAT_C), 1),
        _partial_stats_kernel,
        (x_2d, partial_sum, partial_sum2, BLOCK_STAT_R, BLOCK_STAT_C),
    )

    sum_x = partial_sum.sum(dim=0)
    sum_x2 = partial_sum2.sum(dim=0)
    inv_e = 1.0 / E
    mean_1d = sum_x * inv_e
    var_1d = (sum_x2 * inv_e) - (mean_1d * mean_1d)
    var_1d = torch.clamp(var_1d, min=0.0)
    invstd_1d = torch.rsqrt(var_1d + EPS)

    new_rmean = mean_1d * 0.1 + running_mean * 0.9
    new_rvar = var_1d * CORRECTION * 0.1 + running_var * 0.9
    torch.ops.aten.copy_(running_mean, new_rmean)
    torch.ops.aten.copy_(running_var, new_rvar)

    cat_channels = 2 * C
    # cat storage is NHWC-contig with 2*C channels.
    cat_nhwc = torch.empty((N, HW, cat_channels), device=device, dtype=torch.bfloat16)
    spatial_mean_nc = torch.empty((N, cat_channels), device=device, dtype=torch.bfloat16)

    # Pick BLOCK_C that divides C evenly (so C // BLOCK_C * BLOCK_C == C, and
    # the second store lands at exactly the C-boundary). C is not always a power
    # of 2 (e.g. 240, 336, 480, 60) — find largest divisor <= 32.
    BLOCK_C_kernel = 1
    for cand in (32, 16, 8, 4, 2, 1):
        if C % cand == 0:
            BLOCK_C_kernel = cand
            break
    BLOCK_HW = _pow2_ceil(HW)

    ct.launch(
        stream,
        (N, C // BLOCK_C_kernel, 1),
        _affine_relu_cat_mean_kernel,
        (x_3d, residual_3d, mean_1d, invstd_1d, weight, bias,
         cat_nhwc, spatial_mean_nc,
         HW, BLOCK_HW, BLOCK_C_kernel, C),
    )

    # Materialize NCHW-cat output with the expected channels-last stride:
    # eager cat shape [N, 2*C, H, W] with stride (2*C*HW, 1, W*2*C, 2*C).
    cat_out = torch.empty_strided(
        (N, cat_channels, H, W),
        (cat_channels * HW, 1, W * cat_channels, cat_channels),
        device=device,
        dtype=torch.bfloat16,
    )
    cat_out.copy_(cat_nhwc.view(N, H, W, cat_channels).permute(0, 3, 1, 2))

    spatial_mean = torch.empty_strided(
        (N, cat_channels, 1, 1),
        (cat_channels, 1, 1, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    spatial_mean.view(N, cat_channels).copy_(spatial_mean_nc)

    mean_1x1x1 = mean_1d.view(1, C, 1, 1)
    invstd_1x1x1 = invstd_1d.view(1, C, 1, 1)
    return mean_1x1x1, invstd_1x1x1, cat_out, spatial_mean, running_mean, running_var


@oracle_impl(hardware="B200", point="d3838873")
@oracle_impl(hardware="B200", point="c9e11f4c")
@oracle_impl(hardware="B200", point="0979feae")
@oracle_impl(hardware="B200", point="a58059f1")
def oracle_forward(inputs):
    return _forward(inputs)
