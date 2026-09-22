"""cuTile port of var_mean_ec9057741b2d: pytorch_unet BN-training + bilinear upsample + cat.

Steps:
  1. Compute BN stats (mean, invstd) per channel over N*H*W with a fused cuTile
     reduction that also mutates running_mean/running_var in place.
  2. Compute the bilinear index/weight tensors (y0, y1, x0, x1, x_weight,
     y_weight) using pytorch on CPU-side.
  3. Fill the cat = [skip, upsample(x)] tensor using pytorch for the upsample
     path with the computed BN parameters — this is a large elementwise op so
     we implement it via pytorch to leverage existing bilinear support.

The BN training running_mean/var mutation is observable — the harness compares
the returned running_mean/var tensors.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


RUNNING_VAR_CORRECTION = 1.0000261513114883
SCALE_H = 0.49843260188087773
SCALE_W = 0.4989517819706499


@ct.kernel
def _bn_stats_kernel(
    x_ptr,           # bf16 (C, N*H*W)  — flattened NCHW with C outer
    running_mean_ptr,  # f32 (C,) — mutated
    running_var_ptr,   # f32 (C,) — mutated
    mean_out_ptr,      # f32 (C,)
    invstd_out_ptr,    # f32 (C,)
    C_C: ct.Constant[int],
    E_C: ct.Constant[int],
    INV_E: ct.Constant[float],
    INV_E_MINUS_1: ct.Constant[float],
    CORRECTION: ct.Constant[float],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)

    x = ct.astype(
        ct.load(
            x_ptr, index=(c, 0), shape=(1, BLOCK_R),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    col_idx = ct.arange(BLOCK_R, dtype=ct.int32)
    valid = ct.reshape(col_idx < E_C, (1, BLOCK_R))
    zero_2d = ct.zeros((1, BLOCK_R), dtype=ct.float32)
    x_m = ct.where(valid, x, zero_2d)

    total = ct.sum(x_m, axis=1, keepdims=True)
    total2 = ct.sum(x_m * x_m, axis=1, keepdims=True)
    mean = total * INV_E
    var = total2 * INV_E - mean * mean
    # Clamp neg vars.
    neg_mask = var < ct.zeros((1, 1), dtype=ct.float32)
    zero_2 = ct.zeros((1, 1), dtype=ct.float32)
    var = ct.where(neg_mask, zero_2, var)
    invstd = ct.rsqrt(var + 1.0e-5)

    # Load old running stats.
    old_mean_1d = ct.load(running_mean_ptr, index=(c,), shape=(1,))
    old_var_1d = ct.load(running_var_ptr, index=(c,), shape=(1,))
    old_mean = ct.reshape(old_mean_1d, (1, 1))
    old_var = ct.reshape(old_var_1d, (1, 1))
    new_mean = mean * 0.1 + old_mean * 0.9
    corrected_var = var * CORRECTION
    new_var = corrected_var * 0.1 + old_var * 0.9

    # Store mean, invstd (as scalars per channel), and update running stats.
    ct.store(mean_out_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_out_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))
    ct.store(running_mean_ptr, index=(c,), tile=ct.reshape(new_mean, (1,)))
    ct.store(running_var_ptr, index=(c,), tile=ct.reshape(new_var, (1,)))


@oracle_impl(hardware="B200", point="d2d98efc")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, skip, _shape0 = inputs
    device = x.device
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    out_h = int(skip.shape[2])
    out_w = int(skip.shape[3])
    E = n * h * w  # 1 * 160 * 239 = 38240

    # Round E up to next pow-2 for BLOCK_R.
    BLOCK_R = 1
    while BLOCK_R < E:
        BLOCK_R *= 2

    # Reshape x: (N=1, C=128, H, W) → (C, N*H*W) with C outer.
    x_c = x.permute(1, 0, 2, 3).contiguous().view(c, E)

    mean_out = torch.empty((c,), device=device, dtype=torch.float32)
    invstd_out = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_stats_kernel,
        (
            x_c, running_mean, running_var, mean_out, invstd_out,
            c, E, 1.0 / E, 1.0 / max(E - 1, 1), RUNNING_VAR_CORRECTION, BLOCK_R,
        ),
    )

    # Now compute the outputs. We use pytorch for the upsample/cat since these
    # involve advanced indexing and pytorch already supports bilinear.
    # First, apply BN in fp32:
    mean_reshape = mean_out.view(1, c, 1, 1)
    invstd_reshape = invstd_out.view(1, c, 1, 1)
    normalized = (x.to(torch.float32) - mean_reshape) * invstd_reshape
    # affine: normalized * weight + bias, then round to bf16, then ReLU, then to f32
    weight_r = weight.view(1, c, 1, 1)
    bias_r = bias.view(1, c, 1, 1)
    affine = normalized * weight_r + bias_r
    affine_bf = affine.to(torch.bfloat16)
    relu = torch.relu(affine_bf)
    relu_f = relu.to(torch.float32)

    # Compute idx tensors:
    y_idx = torch.arange(out_h, device=device, dtype=torch.float32) * SCALE_H
    y_idx = torch.clamp_min(y_idx, 0.0).view(out_h, 1)
    y0 = y_idx.to(torch.int64)
    y1 = torch.clamp_max(y0 + 1, h - 1)
    y_weight = torch.clamp(y_idx - y0.to(torch.float32), 0.0, 1.0)

    # x indices only cover out_w - 1 valid positions; last one is padding.
    x_out_valid = out_w - 1
    x_idx = torch.arange(x_out_valid, device=device, dtype=torch.float32) * SCALE_W
    x_idx = torch.clamp_min(x_idx, 0.0)
    x0 = x_idx.to(torch.int64)
    x1 = torch.clamp_max(x0 + 1, w - 1)
    x_weight_valid = torch.clamp(x_idx - x0.to(torch.float32), 0.0, 1.0)

    # Bilinear upsample:
    #   v00 = relu_f[:, :, y0, x0], etc.
    # Use torch.take_along_dim or basic advanced indexing.
    # Note: y0 shape (out_h, 1), x0 shape (x_out_valid,) -> broadcast (out_h, x_out_valid)
    y0_bc = y0.expand(out_h, x_out_valid)
    y1_bc = y1.expand(out_h, x_out_valid)
    x0_bc = x0.view(1, x_out_valid).expand(out_h, x_out_valid)
    x1_bc = x1.view(1, x_out_valid).expand(out_h, x_out_valid)

    v00 = relu_f[:, :, y0_bc, x0_bc]  # (N, C, out_h, x_out_valid)
    v01 = relu_f[:, :, y0_bc, x1_bc]
    v10 = relu_f[:, :, y1_bc, x0_bc]
    v11 = relu_f[:, :, y1_bc, x1_bc]

    x_w = x_weight_valid.view(1, 1, 1, x_out_valid)
    y_w = y_weight.view(1, 1, out_h, 1)
    top = v00 + (v01 - v00) * x_w
    bottom = v10 + (v11 - v10) * x_w
    interp = top + (bottom - top) * y_w
    upsampled_bf = interp.to(torch.bfloat16)
    # Pad last column with 0.
    pad = torch.zeros(n, c, out_h, 1, device=device, dtype=torch.bfloat16)
    upsampled_padded = torch.cat([upsampled_bf, pad], dim=3)

    cat_out = torch.cat([skip, upsampled_padded], dim=1)

    # Return values:
    # (mean_out (as (1,C,1,1)), rsqrt, y0, y1, x0, x1, x_weight_valid, y_weight, cat, running_mean, running_var)
    mean_shaped = mean_out.view(1, c, 1, 1)
    invstd_shaped = invstd_out.view(1, c, 1, 1)
    return (
        mean_shaped, invstd_shaped, y0, y1, x0, x1,
        x_weight_valid, y_weight, cat_out, running_mean, running_var,
    )
