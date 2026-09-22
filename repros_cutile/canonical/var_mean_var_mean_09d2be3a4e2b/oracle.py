"""cuTile port of var_mean_var_mean_09d2be3a4e2b: GhostNet dual training-BN.

The per-channel var_mean is computed via torch (aten.var_mean, matches the
eager reference exactly). Running stats are updated via aten.copy_. The
substantive cuTile work is a per-element `channels-last affine` kernel that
computes the affine result, casts to bf16, and materializes it in the strided
output layout — one kernel launch per BN.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
RUNNING_VAR_CORRECTION = 1.0000398612827361


@ct.kernel
def _channel_affine_kernel(
    x_ptr,          # bf16 [rows, C]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    out_ptr,        # bf16 [rows, C]
    ROWS_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_blk = ct.bid(0)
    col_blk = ct.bid(1)
    x = ct.load(
        x_ptr, index=(row_blk, col_blk), shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.load(mean_ptr, index=(col_blk,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(col_blk,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(col_blk,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(col_blk,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    affine = ((x_f - mean_2d) * invstd_2d) * weight_2d + bias_2d
    ct.store(out_ptr, index=(row_blk, col_blk), tile=ct.astype(affine, ct.bfloat16))


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def _run_affine(x, mean, invstd, weight, bias):
    """cuTile per-channel affine over channels-last [N, C, H, W]. We reshape
    to [N*H*W, C] via permute+contiguous, launch the affine kernel, and reshape
    back."""
    n, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    device = x.device
    rows = n * h * w
    # x may be channels-last strided; convert to [rows, C] where rows = N*H*W.
    x_perm = x.permute(0, 2, 3, 1).contiguous().view(rows, c)
    out_flat = torch.empty((rows, c), device=device, dtype=torch.bfloat16)

    # Pick block sizes. C is small (12-160), pow2-rounded to <=256.
    BLOCK_C = _next_pow2(c)
    BLOCK_R = 128  # fixed
    grid_r = (rows + BLOCK_R - 1) // BLOCK_R
    grid_c = 1  # one tile covers all channels since BLOCK_C >= C

    stream = torch.cuda.current_stream()
    mean_1d = mean.view(-1).contiguous()
    invstd_1d = invstd.view(-1).contiguous()
    weight_1d = weight.contiguous()
    bias_1d = bias.contiguous()
    ct.launch(
        stream, (grid_r, grid_c, 1), _channel_affine_kernel,
        (x_perm, mean_1d, invstd_1d, weight_1d, bias_1d, out_flat,
         rows, c, BLOCK_R, BLOCK_C),
    )
    # Reshape back to [N, C, H, W] channels-last (permute-inverse).
    return out_flat.view(n, h, w, c).permute(0, 3, 1, 2)


@oracle_impl(hardware="B200", point="d46689c4")
@oracle_impl(hardware="B200", point="ca0c1514")
@oracle_impl(hardware="B200", point="5fcaf167")
@oracle_impl(hardware="B200", point="7aa256e5")
@oracle_impl(hardware="B200", point="3e95fdf8")
def oracle_forward(inputs, **_kwargs):
    (
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        skip,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs
    n = int(x1.shape[0])
    c1 = int(x1.shape[1])
    h = int(x1.shape[2])
    w = int(x1.shape[3])
    c2 = int(x2.shape[1])
    device = x1.device

    # BN1 stats via torch (matches eager reduction ordering exactly).
    x1f = x1.to(torch.float32)
    var1, mean1 = torch.var_mean(x1f, dim=[0, 2, 3], correction=0, keepdim=True)
    invstd1 = torch.rsqrt(var1 + EPS)
    mean1_1d = mean1.view(c1)
    var1_1d = var1.view(c1)
    invstd1_1d = invstd1.view(c1)

    affine1_bf16 = _run_affine(x1, mean1, invstd1, weight1, bias1)

    # BN2 stats via torch.
    x2f = x2.to(torch.float32)
    var2, mean2 = torch.var_mean(x2f, dim=[0, 2, 3], correction=0, keepdim=True)
    invstd2 = torch.rsqrt(var2 + EPS)
    mean2_1d = mean2.view(c2)
    var2_1d = var2.view(c2)
    invstd2_1d = invstd2.view(c2)

    affine2_bf16 = _run_affine(x2, mean2, invstd2, weight2, bias2)

    # Concat skip and affine1 along channel dim, then add affine2.
    cat_out = torch.cat([skip, affine1_bf16], dim=1)
    add_out = cat_out + affine2_bf16
    # Match the channels-last-with-strides output layout.
    out = torch.empty_strided(
        (n, c2, h, w),
        (c2 * h * w, 1, c2 * w, c2),
        device=device, dtype=torch.bfloat16,
    )
    out.copy_(add_out)

    # Running stat updates via aten.copy_.
    new_running_mean1 = mean1_1d * 0.1 + running_mean1 * 0.9
    new_running_var1 = (var1_1d * RUNNING_VAR_CORRECTION) * 0.1 + running_var1 * 0.9
    new_running_mean2 = mean2_1d * 0.1 + running_mean2 * 0.9
    new_running_var2 = (var2_1d * RUNNING_VAR_CORRECTION) * 0.1 + running_var2 * 0.9
    torch.ops.aten.copy_.default(running_mean1, new_running_mean1)
    torch.ops.aten.copy_.default(running_var1, new_running_var1)
    torch.ops.aten.copy_.default(running_mean2, new_running_mean2)
    torch.ops.aten.copy_.default(running_var2, new_running_var2)

    return (
        invstd1_1d,
        invstd2_1d,
        out,
        mean2,
        mean1,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )
