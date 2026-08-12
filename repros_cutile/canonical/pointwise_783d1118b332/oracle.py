"""cuTile port of pointwise_783d1118b332: UNet BN + bilinear upsample + cat.

For each output element, computes the branch value via BN+ReLU+bilinear interp,
concatenates with the skip tensor. cuTile IS RTNE by default, so no PTX asm.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 512
IN_H = 40
IN_W = 59
OUT_H = 80
OUT_W = 119
UPSAMPLE_W = 118


@ct.kernel
def _upsample_cat_kernel(
    mean_ptr,       # bf16 [C]
    x_ptr,          # bf16 [1, C, IN_H, IN_W]
    var_ptr,        # bf16 [C]
    weight_ptr,     # bf16 [C]
    bias_ptr,       # bf16 [C]
    skip_ptr,       # bf16 [1, C, OUT_H, OUT_W]
    out_ptr,        # bf16 [TOTAL] flattened
    TOTAL: ct.Constant[int],
    C_C: ct.Constant[int],
    IN_H_C: ct.Constant[int],
    IN_W_C: ct.Constant[int],
    OUT_H_C: ct.Constant[int],
    OUT_W_C: ct.Constant[int],
    UPSAMPLE_W_C: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK_E, dtype=ct.int64) + pid * BLOCK_E

    ow = offsets - (offsets // OUT_W_C) * OUT_W_C
    t0 = offsets // OUT_W_C
    oh = t0 - (t0 // OUT_H_C) * OUT_H_C
    channel = (t0 // OUT_H_C) - ((t0 // OUT_H_C) // (C_C + C_C)) * (C_C + C_C)
    batch = t0 // (OUT_H_C * (C_C + C_C))
    zero_i = ct.zeros((BLOCK_E,), dtype=ct.int64)
    C_C_tile = ct.full((BLOCK_E,), C_C, dtype=ct.int64)
    first_half = channel < C_C_tile
    branch_channel = channel - C_C_tile

    # Skip loads (only valid when first_half)
    skip_channel = ct.where(first_half, channel, zero_i)
    skip_offsets = batch * (C_C * OUT_H_C * OUT_W_C) + skip_channel * (OUT_H_C * OUT_W_C) + oh * OUT_W_C + ow
    skip_value = ct.gather(skip_ptr, skip_offsets)

    # Branch computation (only valid when !first_half and ow < UPSAMPLE_W)
    ow_f = ct.astype(ow, ct.float32)
    oh_f = ct.astype(oh, ct.float32)
    row_f = oh_f * 0.4936708860759494
    zero_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    row_f = ct.where(row_f > zero_f, row_f, zero_f)
    row0 = ct.astype(row_f, ct.int64)
    row0_f = ct.astype(row0, ct.float32)
    one_i = ct.full((BLOCK_E,), 1, dtype=ct.int64)
    IN_H_i = ct.full((BLOCK_E,), IN_H_C - 1, dtype=ct.int64)
    row1_maybe = row0 + one_i
    row1 = ct.where(row1_maybe < IN_H_i, row1_maybe, IN_H_i)
    row_alpha = row_f - row0_f
    one_f = ct.full((BLOCK_E,), 1.0, dtype=ct.float32)
    row_alpha_low = ct.where(row_alpha > zero_f, row_alpha, zero_f)
    row_alpha = ct.where(row_alpha_low < one_f, row_alpha_low, one_f)

    col_f = ow_f * 0.49572649572649574
    col_f = ct.where(col_f > zero_f, col_f, zero_f)
    col0 = ct.astype(col_f, ct.int64)
    col0_f = ct.astype(col0, ct.float32)
    IN_W_i = ct.full((BLOCK_E,), IN_W_C - 1, dtype=ct.int64)
    col1_maybe = col0 + one_i
    col1 = ct.where(col1_maybe < IN_W_i, col1_maybe, IN_W_i)
    col_alpha = col_f - col0_f
    col_alpha_low = ct.where(col_alpha > zero_f, col_alpha, zero_f)
    col_alpha = ct.where(col_alpha_low < one_f, col_alpha_low, one_f)

    # BN + ReLU on-the-fly for 4 pixel locations
    # Safe branch_channel: clip to [0, C)
    safe_bc = ct.where(first_half, zero_i, branch_channel)
    mean = ct.astype(ct.gather(mean_ptr, safe_bc), ct.float32)
    var = ct.astype(ct.gather(var_ptr, safe_bc), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, safe_bc), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, safe_bc), ct.float32)

    denom = ct.sqrt(var + 1.0e-5)
    invstd = 1.0 / denom

    input_base = batch * (C_C * IN_H_C * IN_W_C) + safe_bc * (IN_H_C * IN_W_C)

    def _bn_value(row_ii, col_ii):
        x_bf = ct.gather(x_ptr, input_base + row_ii * IN_W_C + col_ii)
        x = ct.astype(x_bf, ct.float32)
        centered = x - mean
        norm = centered * invstd
        affine = norm * weight + bias
        rounded_bf = ct.astype(affine, ct.bfloat16)
        rounded = ct.astype(rounded_bf, ct.float32)
        # ReLU preserving NaN: (x > 0) | (x != x) ? x : 0
        is_pos = rounded > zero_f
        is_nan = rounded != rounded
        keep = is_pos | is_nan
        return ct.where(keep, rounded, zero_f)

    v00 = _bn_value(row0, col0)
    v01 = _bn_value(row0, col1)
    v10 = _bn_value(row1, col0)
    v11 = _bn_value(row1, col1)

    top = v00 + (v01 - v00) * col_alpha
    bottom = v10 + (v11 - v10) * col_alpha
    interp = top + (bottom - top) * row_alpha
    interp_bf = ct.astype(interp, ct.bfloat16)

    # For invalid branch positions (ow >= UPSAMPLE_W), branch value = 0.0
    UPSAMPLE_W_i = ct.full((BLOCK_E,), UPSAMPLE_W_C, dtype=ct.int64)
    valid_branch_ow = ow < UPSAMPLE_W_i
    zero_bf = ct.zeros((BLOCK_E,), dtype=ct.bfloat16)
    branch_value = ct.where(valid_branch_ow, interp_bf, zero_bf)

    value = ct.where(first_half, skip_value, branch_value)
    # Mask writes to only valid indices
    valid = offsets < ct.full((BLOCK_E,), TOTAL, dtype=ct.int64)
    ct.scatter(out_ptr, offsets, value, mask=valid)


@oracle_impl(hardware="B200", point="a2b68844", BLOCK_E=256)
def oracle_forward(inputs, *, BLOCK_E: int):
    mean, x, var, weight, bias, skip, _shape_param_0 = inputs

    channels = C
    out = torch.empty_strided(
        (1, channels + channels, OUT_H, OUT_W),
        ((channels + channels) * OUT_H * OUT_W, OUT_H * OUT_W, OUT_W, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    total = out.numel()
    out_flat = out.view(total)
    x_flat = x.contiguous().view(-1)
    skip_flat = skip.contiguous().view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK_E), 1, 1),
        _upsample_cat_kernel,
        (mean, x_flat, var, weight, bias, skip_flat, out_flat,
         total, C, IN_H, IN_W, OUT_H, OUT_W, UPSAMPLE_W, BLOCK_E),
    )
    return out
