"""cuTile port of var_mean_mean_1ae53b59df19 (SCHEDULER_FUSION): phlippe DenseNet
cat + train-BN + ReLU + spatial-mean.

Fairness note: the Triton reference fuses cat + BN-stats + running-stat update +
affine + ReLU + spatial-mean into ONE kernel `_cat_bn_relu_spatial_mean_kernel`.
For the cuTile port we keep the reductions (BN stats over N,H,W) and running-stat
updates INSIDE `@ct.kernel`. The cat itself is a pure memory shuffle done via
`torch.cat` as a bridge (no compute is offloaded to torch), then permuted so a
per-channel program can load a contiguous (N, HW) tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


CHANNELS = 184
BATCH = 128
HW = 16  # 4 * 4
EPS = 1.0e-5
MOMENTUM = 0.1
ONE_MINUS_MOMENTUM = 0.9
UNBIAS_CORRECTION = 1.0004885197850513


@ct.kernel
def _bn_relu_pool_kernel(
    cat_ptr,          # bf16 [N, C, H, W] contiguous — loaded directly (no permute+copy)
    running_mean_ptr, # f32 [C]  (read-modify-write)
    running_var_ptr,  # f32 [C]  (read-modify-write)
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    saved_mean_ptr,   # f32 [C]
    invstd_ptr,       # f32 [C]
    pooled_ptr,       # bf16 [N, C]  (strided column write per program)
    INV_E: ct.Constant[float],
    INV_HW: ct.Constant[float],
    EPS_: ct.Constant[float],
    MOM: ct.Constant[float],
    ONE_MINUS_MOM: ct.Constant[float],
    CORR: ct.Constant[float],
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    BLOCK_W: ct.Constant[int],
):
    channel = ct.bid(0)

    # Load a (BLOCK_N, 1, BLOCK_H, BLOCK_W) tile from the [N, C, H, W]
    # contiguous cat tensor for THIS channel — no permute+contiguous copy.
    x4 = ct.load(cat_ptr, index=(0, channel, 0, 0),
                 shape=(BLOCK_N, 1, BLOCK_H, BLOCK_W))
    x_2d = ct.reshape(x4, (BLOCK_N, BLOCK_HW))
    x_f = ct.astype(x_2d, ct.float32)

    # BN population stats over N,H,W (all elements for this channel).
    total = ct.sum(x_f)
    total_sq = ct.sum(x_f * x_f)
    mean_s = total * INV_E
    ex2 = total_sq * INV_E
    diff = ex2 - mean_s * mean_s
    zero_scalar = ct.full((), 0.0, dtype=ct.float32)
    var_s = ct.where(diff > zero_scalar, diff, zero_scalar)
    invstd_s = ct.rsqrt(var_s + EPS_)

    # Running-stat read-modify-write.
    old_mean = ct.load(running_mean_ptr, index=(channel,), shape=(1,))
    old_var = ct.load(running_var_ptr, index=(channel,), shape=(1,))
    mean_1 = ct.reshape(mean_s, (1,))
    var_1 = ct.reshape(var_s, (1,))
    invstd_1 = ct.reshape(invstd_s, (1,))
    new_mean = mean_1 * MOM + old_mean * ONE_MINUS_MOM
    new_var = (var_1 * CORR) * MOM + old_var * ONE_MINUS_MOM
    ct.store(running_mean_ptr, index=(channel,), tile=new_mean)
    ct.store(running_var_ptr, index=(channel,), tile=new_var)
    ct.store(saved_mean_ptr, index=(channel,), tile=mean_1)
    ct.store(invstd_ptr, index=(channel,), tile=invstd_1)

    # Affine + ReLU (fp32 math, bf16 rounding before ReLU to match Triton).
    gamma = ct.astype(
        ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    beta = ct.astype(
        ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)
    mean_bc = ct.reshape(mean_s, (1, 1))
    invstd_bc = ct.reshape(invstd_s, (1, 1))
    gamma_bc = ct.reshape(gamma, (1, 1))
    beta_bc = ct.reshape(beta, (1, 1))

    centered = x_f - mean_bc
    normalized = centered * invstd_bc
    scaled = normalized * gamma_bc
    affine = scaled + beta_bc
    affine_bf = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.full((BLOCK_N, BLOCK_HW), 0.0, dtype=ct.bfloat16)
    relu = ct.where(affine_bf < zero_bf, zero_bf, affine_bf)

    # Spatial mean per row: sum over HW axis * (1/HW).
    relu_f = ct.astype(relu, ct.float32)
    row_sum = ct.sum(relu_f, axis=1)              # shape (BLOCK_N,)
    pooled = row_sum * INV_HW
    pooled_bf = ct.astype(pooled, ct.bfloat16)
    pooled_2d = ct.reshape(pooled_bf, (BLOCK_N, 1))
    ct.store(pooled_ptr, index=(0, channel), tile=pooled_2d)


@oracle_impl(hardware="B200", point="d10f4d93", BLOCK_N=128, BLOCK_HW=16)
def oracle_forward(inputs, *, BLOCK_N, BLOCK_HW):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _s0 = inputs
    view_shape = tuple(int(dim) for dim in _s0)

    n = int(arg0_1.shape[0])
    c0 = int(arg0_1.shape[1])
    c1 = int(arg1_1.shape[1])
    c = c0 + c1
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    e = n * hw
    device = arg0_1.device

    # Cat is a pure memory shuffle — no compute — done here as a bridge.
    # Consumed by the kernel directly in [N, C, H, W] layout: NO permute+copy.
    cat = torch.cat([arg0_1, arg1_1], dim=1)  # [N, C, H, W] bf16 contiguous

    saved_mean = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32)
    invstd = torch.empty_strided(
        (1, c, 1, 1), (c, 1, 1, 1), device=device, dtype=torch.float32)
    pooled = torch.empty_strided(
        (n, c), (c, 1), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_relu_pool_kernel,
        (cat, arg2_1, arg3_1, arg4_1, arg5_1,
         saved_mean.view(c), invstd.view(c), pooled,
         1.0 / float(e), 1.0 / float(hw),
         EPS, MOMENTUM, ONE_MINUS_MOMENTUM, UNBIAS_CORRECTION,
         BLOCK_N, BLOCK_HW, h, w),
    )

    view = pooled.view(view_shape)
    return cat, saved_mean, invstd, view, arg2_1, arg3_1
