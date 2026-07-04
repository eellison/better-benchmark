"""cuTile port of var_mean_mean_1f7174490c38: MobileNetV2 BN training + ReLU6 + spatial mean.

Two cuTile kernels mirroring the Triton reference (no torch reductions/math
in oracle_forward beyond the necessary channels-last permute/reshape and the
HW zero-pad required to tile the epilogue cleanly).

1. _bn_stats_kernel: per-channel reduction over N*H*W in a single pass,
   computing saved_mean, invstd, and updating running_mean/running_var
   IN-KERNEL via the captured momentum + unbiased-variance correction literal.
2. _relu6_spatial_mean_kernel: fp32 affine + bf16 round + ReLU6 + spatial
   mean, writing bf16 [N, C].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
CORRECTION = 1.0001594642002871
MOMENTUM = 0.1
ONE_MINUS_MOMENTUM = 0.9


@ct.kernel
def _bn_stats_kernel(
    x_ptr,             # bf16 [E, C]
    running_mean_ptr,  # f32 [C]  (read-modify-write)
    running_var_ptr,   # f32 [C]  (read-modify-write)
    saved_mean_ptr,    # f32 [C]
    invstd_ptr,        # f32 [C]
    E: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    INV_E: ct.Constant[float],
    EPS_: ct.Constant[float],
    CORR: ct.Constant[float],
):
    cblk = ct.bid(0)

    x = ct.load(
        x_ptr, index=(0, cblk), shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    # OOB rows (BLOCK_R > E) are already 0.0 thanks to PaddingMode.ZERO, so
    # they contribute nothing to the population sum/sum-of-squares.
    sums = ct.sum(x_f, axis=0)             # (BLOCK_C,)
    sums2 = ct.sum(x_f * x_f, axis=0)      # (BLOCK_C,)
    mean = sums * INV_E
    ex2 = sums2 * INV_E
    var_raw = ex2 - mean * mean
    zero_c = ct.full((BLOCK_C,), 0.0, dtype=ct.float32)
    var = ct.where(var_raw < zero_c, zero_c, var_raw)
    invstd = ct.rsqrt(var + EPS_)

    old_mean = ct.load(running_mean_ptr, index=(cblk,), shape=(BLOCK_C,))
    old_var = ct.load(running_var_ptr, index=(cblk,), shape=(BLOCK_C,))
    new_mean = mean * 0.1 + old_mean * 0.9
    new_var = (var * CORR) * 0.1 + old_var * 0.9

    ct.store(saved_mean_ptr, index=(cblk,), tile=mean)
    ct.store(invstd_ptr, index=(cblk,), tile=invstd)
    ct.store(running_mean_ptr, index=(cblk,), tile=new_mean)
    ct.store(running_var_ptr, index=(cblk,), tile=new_var)


@ct.kernel
def _relu6_spatial_mean_kernel(
    x_ptr,           # bf16 [N*BLOCK_HW, C]  (HW padded with zeros to BLOCK_HW)
    saved_mean_ptr,  # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    out_ptr,         # bf16 [N, C]
    HW: ct.Constant[int],
    INV_HW: ct.Constant[float],
    BLOCK_HW: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    n = ct.bid(0)
    cblk = ct.bid(1)

    x = ct.load(x_ptr, index=(n, cblk), shape=(BLOCK_HW, ROW_BLOCK))
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(saved_mean_ptr, index=(cblk,), shape=(ROW_BLOCK,))
    invstd = ct.load(invstd_ptr, index=(cblk,), shape=(ROW_BLOCK,))
    weight = ct.load(weight_ptr, index=(cblk,), shape=(ROW_BLOCK,))
    bias = ct.load(bias_ptr, index=(cblk,), shape=(ROW_BLOCK,))
    mean_2 = ct.reshape(mean, (1, ROW_BLOCK))
    invstd_2 = ct.reshape(invstd, (1, ROW_BLOCK))
    weight_2 = ct.reshape(weight, (1, ROW_BLOCK))
    bias_2 = ct.reshape(bias, (1, ROW_BLOCK))

    normalized = (x_f - mean_2) * invstd_2
    affine = normalized * weight_2 + bias_2
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_rounded = ct.astype(affine_bf, ct.float32)

    zero_f = ct.full((BLOCK_HW, ROW_BLOCK), 0.0, dtype=ct.float32)
    six_f = ct.full((BLOCK_HW, ROW_BLOCK), 6.0, dtype=ct.float32)
    clamped_min = ct.where(affine_rounded < zero_f, zero_f, affine_rounded)
    clamped = ct.where(clamped_min > six_f, six_f, clamped_min)
    relu6_bf = ct.astype(clamped, ct.bfloat16)
    relu6_f = ct.astype(relu6_bf, ct.float32)

    # Mask HW-padded slots so they contribute 0 to the spatial sum.
    hw_arange = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_valid = hw_arange < HW
    hw_valid_2 = ct.reshape(hw_valid, (BLOCK_HW, 1))
    relu6_masked = ct.where(hw_valid_2, relu6_f, zero_f)

    pooled = ct.sum(relu6_masked, axis=0, keepdims=True) * INV_HW  # (1, ROW_BLOCK)
    ct.store(out_ptr, index=(n, cblk), tile=ct.astype(pooled, ct.bfloat16))


@oracle_impl(
    hardware="B200", point="86b98b1a",
    BLOCK_R=8192, BLOCK_C=16, BLOCK_HW=64, ROW_BLOCK=64,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    ROW_BLOCK: int,
):
    x, running_mean, running_var, weight, bias, _sp0, _sp1, _sp2 = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    e = n * hw
    device = x.device

    # x is channels-last [N,C,H,W] (memory layout is NHWC-contiguous).
    # permute(0,2,3,1) yields a contiguous [N,H,W,C] view; reshape to [E, C].
    x_flat = x.permute(0, 2, 3, 1).reshape(e, c)  # bf16 [E, C], contiguous

    saved_mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32,
    )
    saved_mean_1d = saved_mean.view(c)
    invstd_1d = invstd.view(c)

    stream = torch.cuda.current_stream()

    # Kernel 1: per-channel BN stats + in-kernel running-stat update.
    ct.launch(
        stream,
        (c // BLOCK_C, 1, 1),
        _bn_stats_kernel,
        (
            x_flat, running_mean, running_var,
            saved_mean_1d, invstd_1d,
            e, BLOCK_R, BLOCK_C,
            1.0 / float(e), EPS, CORRECTION,
        ),
    )

    # Pad the HW dim to BLOCK_HW so the epilogue tiles cleanly. Uses NHWC.
    padded_x = torch.zeros((n, BLOCK_HW, c), device=device, dtype=torch.bfloat16)
    padded_x[:, :hw, :].copy_(x_flat.view(n, hw, c))
    padded_x_2d = padded_x.view(n * BLOCK_HW, c)

    spatial_mean = torch.empty_strided(
        (n, c), (c, 1), device=device, dtype=torch.bfloat16,
    )

    # Kernel 2: affine + bf16 round + ReLU6 + spatial mean.
    ct.launch(
        stream,
        (n, c // ROW_BLOCK, 1),
        _relu6_spatial_mean_kernel,
        (
            padded_x_2d, saved_mean_1d, invstd_1d, weight, bias,
            spatial_mean,
            hw, 1.0 / float(hw), BLOCK_HW, ROW_BLOCK,
        ),
    )

    return saved_mean, invstd, spatial_mean, running_mean, running_var
