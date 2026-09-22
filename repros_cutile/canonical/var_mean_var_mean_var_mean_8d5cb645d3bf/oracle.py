"""cuTile port of var_mean_var_mean_var_mean_8d5cb645d3bf: 6-branch Inception BN training.

Var/mean reductions and running-stat updates done in torch. cuTile kernel does
the BN affine + ReLU per branch (channels-last), and a final spatial-mean over
the concatenated output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 8
WIDTH = 8
HW = 64
OUT_CHANNELS = 2048


@ct.kernel
def _bn_relu_affine_kernel(
    x_ptr,           # bf16 [B, HW, C]
    mean_ptr,        # f32  [C]
    invstd_ptr,      # f32  [C]
    weight_ptr,      # f32  [C]
    bias_ptr,        # f32  [C]
    out_ptr,         # bf16 [B, HW, C]
    C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    batch = ct.bid(0)
    hw_block = ct.bid(1)
    c_block = ct.bid(2)

    x_bf = ct.load(x_ptr, index=(batch, hw_block, c_block), shape=(1, BLOCK_HW, BLOCK_C))
    x = ct.astype(x_bf, ct.float32)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_3d = ct.reshape(mean, (1, 1, BLOCK_C))
    invstd_3d = ct.reshape(invstd, (1, 1, BLOCK_C))
    weight_3d = ct.reshape(weight, (1, 1, BLOCK_C))
    bias_3d = ct.reshape(bias, (1, 1, BLOCK_C))

    centered = x - mean_3d
    normed = centered * invstd_3d
    scaled = normed * weight_3d
    affine = scaled + bias_3d
    rounded = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded, ct.float32)
    zero = ct.zeros((1, BLOCK_HW, BLOCK_C), dtype=ct.float32)
    is_pos = rounded_f > 0.0
    is_nan = rounded_f != rounded_f
    keep = is_pos | is_nan
    relu = ct.where(keep, rounded_f, zero)
    out_bf = ct.astype(relu, ct.bfloat16)
    ct.store(out_ptr, index=(batch, hw_block, c_block), tile=out_bf)


@ct.kernel
def _spatial_mean_kernel(
    x_ptr,        # bf16 [B, HW, C]
    out_ptr,      # bf16 [B, C]
    HW_SIZE: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    batch = ct.bid(0)
    c_block = ct.bid(1)

    # Sum bf16 values over HW dim (in fp32), divide by HW.
    x_bf = ct.load(x_ptr, index=(batch, 0, c_block), shape=(1, HW_SIZE, BLOCK_C))
    x = ct.astype(x_bf, ct.float32)
    total = ct.sum(x, axis=1)  # [1, BLOCK_C]
    mean = total * (1.0 / HW_SIZE)
    out_bf = ct.astype(mean, ct.bfloat16)
    out_2d = ct.reshape(out_bf, (1, BLOCK_C))
    ct.store(out_ptr, index=(batch, c_block), tile=out_2d)


def _run_branch(x, running_mean, running_var, weight, bias, *, block_hw, block_c):
    """Per-branch BN training reduction + affine.

    x: bf16 channels-last [B, C, H, W].
    running_mean, running_var, weight, bias: f32 [C].
    Returns: (mean_kept, invstd_kept, relu_output, new_running_mean, new_running_var)
    - mean_kept: f32 [1, C, 1, 1] (getitem_1 shape)
    - invstd_kept: f32 [1, C, 1, 1] (rsqrt shape)
    - relu_output: bf16 [B, C, H, W] channels-last
    - new_running_mean / new_running_var: f32 [C]
    """
    device = x.device
    C = int(x.shape[1])
    # Reduction over [0, 2, 3] (batch + spatial) -> [C]
    x_f32 = x.float()
    mean_c = x_f32.mean(dim=[0, 2, 3])
    var_c = ((x_f32 - mean_c.view(1, C, 1, 1)) ** 2).mean(dim=[0, 2, 3])
    invstd_c = torch.rsqrt(var_c + 0.001)

    # Running-stat update: new = old * 0.9 + stat * 0.1  (var also scaled by 1.00012...)
    new_running_mean = running_mean * 0.9 + mean_c * 0.1
    new_running_var = running_var * 0.9 + (var_c * 1.0001220852154804) * 0.1

    # Affine + ReLU with cuTile
    x_bhwc = x.permute(0, 2, 3, 1).reshape(BATCH, HW, C).contiguous()
    out_bhwc = torch.empty(BATCH, HW, C, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    grid = (BATCH, ct.cdiv(HW, block_hw), ct.cdiv(C, block_c))
    ct.launch(
        stream, grid, _bn_relu_affine_kernel,
        (x_bhwc, mean_c, invstd_c, weight, bias, out_bhwc,
         C, block_hw, block_c),
    )
    # Reshape back to channels-last [B, C, H, W]
    out_bchw = torch.empty_strided(
        (BATCH, C, HEIGHT, WIDTH),
        (C * HW, 1, WIDTH * C, C),
        device=device, dtype=torch.bfloat16)
    out_bchw.copy_(out_bhwc.view(BATCH, HEIGHT, WIDTH, C).permute(0, 3, 1, 2))

    mean_kept = mean_c.view(1, C, 1, 1).contiguous()
    invstd_kept = invstd_c.view(1, C, 1, 1).contiguous()
    return mean_kept, invstd_kept, out_bchw, new_running_mean, new_running_var


@oracle_impl(hardware="B200", point="13e49e96", BLOCK_HW=16, BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int):
    (
        x0, running_mean0, running_var0, weight0, bias0,
        x1, running_mean1, running_var1, weight1, bias1,
        x2, running_mean2, running_var2, weight2, bias2,
        x3, running_mean3, running_var3, weight3, bias3,
        x4, running_mean4, running_var4, weight4, bias4,
        x5, running_mean5, running_var5, weight5, bias5,
        _sp0, _sp1, _sp2,
    ) = inputs
    device = x0.device

    m0, r0, o0, rm0_new, rv0_new = _run_branch(
        x0, running_mean0, running_var0, weight0, bias0,
        block_hw=BLOCK_HW, block_c=BLOCK_C)
    m1, r1, o1, rm1_new, rv1_new = _run_branch(
        x1, running_mean1, running_var1, weight1, bias1,
        block_hw=BLOCK_HW, block_c=BLOCK_C)
    m2, r2, o2, rm2_new, rv2_new = _run_branch(
        x2, running_mean2, running_var2, weight2, bias2,
        block_hw=BLOCK_HW, block_c=BLOCK_C)
    m3, r3, o3, rm3_new, rv3_new = _run_branch(
        x3, running_mean3, running_var3, weight3, bias3,
        block_hw=BLOCK_HW, block_c=BLOCK_C)
    m4, r4, o4, rm4_new, rv4_new = _run_branch(
        x4, running_mean4, running_var4, weight4, bias4,
        block_hw=BLOCK_HW, block_c=BLOCK_C)
    m5, r5, o5, rm5_new, rv5_new = _run_branch(
        x5, running_mean5, running_var5, weight5, bias5,
        block_hw=BLOCK_HW, block_c=BLOCK_C)

    # Copy_ mutations to input running_mean and running_var
    running_mean0.copy_(rm0_new)
    running_var0.copy_(rv0_new)
    running_mean1.copy_(rm1_new)
    running_var1.copy_(rv1_new)
    running_mean2.copy_(rm2_new)
    running_var2.copy_(rv2_new)
    running_mean3.copy_(rm3_new)
    running_var3.copy_(rv3_new)
    running_mean4.copy_(rm4_new)
    running_var4.copy_(rv4_new)
    running_mean5.copy_(rm5_new)
    running_var5.copy_(rv5_new)

    # cat: order in ref is [relu, cat, cat_1, relu_5] where cat=[relu_1, relu_2], cat_1=[relu_3, relu_4]
    # so final cat is [relu, relu_1, relu_2, relu_3, relu_4, relu_5]
    # in our indexing: relu -> o0 (320), relu_1 -> o1 (384), relu_2 -> o2 (384),
    # relu_3 -> o3 (384), relu_4 -> o4 (384), relu_5 -> o5 (192)
    # total = 320+384*4+192 = 2048.
    cat_bchw = torch.cat([o0, o1, o2, o3, o4, o5], dim=1)  # [128, 2048, 8, 8]

    # Spatial mean over [-1, -2]
    # Reference: mean.dim(cat_2, [-1, -2], True) => [128, 2048, 1, 1]
    # We reshape to [B, HW, C] channels-last then do cuTile reduce.
    cat_bhwc = cat_bchw.permute(0, 2, 3, 1).reshape(BATCH, HW, OUT_CHANNELS).contiguous()
    view_out = torch.empty(BATCH, OUT_CHANNELS, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    grid = (BATCH, ct.cdiv(OUT_CHANNELS, BLOCK_C), 1)
    ct.launch(
        stream, grid, _spatial_mean_kernel,
        (cat_bhwc, view_out, HW, OUT_CHANNELS, BLOCK_C),
    )

    return (
        m0, r0, m1, r1, m2, r2, m3, r3, m4, r4, m5, r5,
        view_out,
        running_mean0, running_var0,
        running_mean1, running_var1,
        running_mean2, running_var2,
        running_mean3, running_var3,
        running_mean4, running_var4,
        running_mean5, running_var5,
    )
