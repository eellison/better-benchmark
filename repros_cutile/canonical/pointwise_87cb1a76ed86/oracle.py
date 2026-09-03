"""cuTile port of pointwise_87cb1a76ed86: PyTorch-UNet BN+ReLU+bilinear resize+cat.

Two kernels:
1. Copy skip -> first half of cat output.
2. BN-inference affine + ReLU + bilinear resize + right-pad -> second half.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


CHANNELS = 128
IN_H = 160
IN_W = 239
OUT_H = 320
OUT_W_INTERP = 478
OUT_W = 479
IN_HW = IN_H * IN_W
OUT_HW = OUT_H * OUT_W


@ct.kernel
def _copy_first_half_kernel(
    src_ptr,        # bf16 flat [CHANNELS*OUT_HW]
    out_ptr,        # bf16 flat, offset into first half
    N_ELEMS: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    vals = ct.load(
        src_ptr, index=(pid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO
    )
    # scatter with mask to avoid OOB
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = offsets < N_ELEMS
    ct.scatter(out_ptr, offsets, vals, mask=mask)


@ct.kernel
def _resize_pad_second_half_kernel(
    mean_ptr,       # bf16 [CHANNELS]
    x_ptr,          # bf16 flat [CHANNELS*IN_HW]
    var_ptr,        # bf16 [CHANNELS]
    weight_ptr,     # bf16 [CHANNELS]
    bias_ptr,       # bf16 [CHANNELS]
    out_ptr,        # bf16 flat, offset into second half [CHANNELS*OUT_HW]
    C_: ct.Constant[int],
    IH_: ct.Constant[int],
    IW_: ct.Constant[int],
    OH_: ct.Constant[int],
    OW_INTERP_: ct.Constant[int],
    OW_: ct.Constant[int],
    IHW_: ct.Constant[int],
    OHW_: ct.Constant[int],
    BLOCK_W: ct.Constant[int],
):
    # Grid: (CHANNELS * OH, cdiv(OW, BLOCK_W), 1)
    pid = ct.bid(0)
    w_block = ct.bid(1)

    c = pid // OH_
    oh = pid - c * OH_

    cols = ct.arange(BLOCK_W, dtype=ct.int32)
    ow = w_block * BLOCK_W + cols
    out_mask = ow < OW_
    interp_mask = ow < OW_INTERP_

    # Load per-channel BN parameters (scalar per channel)
    mean_idx = ct.full((1,), c, dtype=ct.int64)
    mean = ct.astype(ct.gather(mean_ptr, mean_idx), ct.float32)
    var = ct.astype(ct.gather(var_ptr, mean_idx), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, mean_idx), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, mean_idx), ct.float32)
    invstd = ct.rsqrt(var + 1.0e-5)

    # Compute src_h once
    src_h_scalar_i32 = oh
    src_h_scalar_f = ct.astype(ct.full((1,), src_h_scalar_i32, dtype=ct.int32), ct.float32) * 0.49843260188087773
    # clamp_min 0.0
    zero_1 = ct.full((1,), 0.0, dtype=ct.float32)
    src_h_scalar_f = ct.where(src_h_scalar_f > zero_1, src_h_scalar_f, zero_1)
    h_lo_1 = ct.astype(src_h_scalar_f, ct.int64)
    h_hi_val = h_lo_1 + 1
    ih_minus1 = ct.full((1,), IH_ - 1, dtype=ct.int64)
    h_hi_1 = ct.where(h_hi_val < ih_minus1, h_hi_val, ih_minus1)
    h_frac_1 = src_h_scalar_f - ct.astype(h_lo_1, ct.float32)
    one_1 = ct.full((1,), 1.0, dtype=ct.float32)
    h_frac_1 = ct.where(h_frac_1 > zero_1, h_frac_1, zero_1)
    h_frac_1 = ct.where(h_frac_1 < one_1, h_frac_1, one_1)

    # Compute src_w tile
    ow_f = ct.astype(ow, ct.float32) * 0.4989517819706499
    zero_w = ct.full((BLOCK_W,), 0.0, dtype=ct.float32)
    src_w = ct.where(ow_f > zero_w, ow_f, zero_w)
    w_lo = ct.astype(src_w, ct.int64)
    w_hi_val = w_lo + 1
    iw_minus1 = ct.full((BLOCK_W,), IW_ - 1, dtype=ct.int64)
    w_hi = ct.where(w_hi_val < iw_minus1, w_hi_val, iw_minus1)
    w_frac = src_w - ct.astype(w_lo, ct.float32)
    one_w = ct.full((BLOCK_W,), 1.0, dtype=ct.float32)
    w_frac = ct.where(w_frac > zero_w, w_frac, zero_w)
    w_frac = ct.where(w_frac < one_w, w_frac, one_w)

    base = c * IHW_
    # Extract scalar h_lo, h_hi as int64. We can broadcast by * IW_ + w_lo.
    # But h_lo_1, h_hi_1 are shape (1,); broadcast against (BLOCK_W,)
    def _bn_relu_sample(offsets_1d, mask_tile):
        # offsets_1d: int64 shape (BLOCK_W,)
        x = ct.astype(ct.gather(x_ptr, offsets_1d), ct.float32)
        centered = x - mean
        normalized = centered * invstd
        scaled = normalized * weight
        affine = scaled + bias
        affine_bf16 = ct.astype(affine, ct.bfloat16)
        zero_bf16 = ct.full((BLOCK_W,), 0.0, dtype=ct.bfloat16)
        relu_bf16 = ct.where(affine_bf16 < zero_bf16, zero_bf16, affine_bf16)
        return ct.astype(relu_bf16, ct.float32)

    # broadcast h_lo/h_hi over BLOCK_W
    h_lo_b = ct.full((BLOCK_W,), 0, dtype=ct.int64) + h_lo_1  # broadcast (1,) -> (BLOCK_W,)
    h_hi_b = ct.full((BLOCK_W,), 0, dtype=ct.int64) + h_hi_1
    base_1d = ct.full((BLOCK_W,), 0, dtype=ct.int64) + ct.full((1,), base, dtype=ct.int64)

    off_hi_hi = base_1d + h_hi_b * IW_ + w_hi
    off_hi_lo = base_1d + h_hi_b * IW_ + w_lo
    off_lo_hi = base_1d + h_lo_b * IW_ + w_hi
    off_lo_lo = base_1d + h_lo_b * IW_ + w_lo

    v_hi_hi = _bn_relu_sample(off_hi_hi, interp_mask)
    v_hi_lo = _bn_relu_sample(off_hi_lo, interp_mask)
    v_lo_hi = _bn_relu_sample(off_lo_hi, interp_mask)
    v_lo_lo = _bn_relu_sample(off_lo_lo, interp_mask)

    hi_row = v_hi_lo + (v_hi_hi - v_hi_lo) * w_frac
    lo_row = v_lo_lo + (v_lo_hi - v_lo_lo) * w_frac
    h_frac_b = ct.full((BLOCK_W,), 0.0, dtype=ct.float32) + h_frac_1
    value = lo_row + (hi_row - lo_row) * h_frac_b
    value = ct.where(interp_mask, value, zero_w)

    # Store to out_ptr at (c, oh, ow), i.e., c*OHW + oh*OW + ow, only if ow < OW.
    out_offsets = c * OHW_ + oh * OW_ + ow
    value_bf16 = ct.astype(value, ct.bfloat16)
    ct.scatter(out_ptr, out_offsets, value_bf16, mask=out_mask)


@oracle_impl(hardware="B200", point="f4799916", COPY_BLOCK=1024, BLOCK_W=512)
def oracle_forward(inputs, *, COPY_BLOCK: int, BLOCK_W: int):
    mean, x, var, weight, bias, skip, _shape = inputs
    device = x.device
    out = torch.empty(
        (1, 2 * CHANNELS, OUT_H, OUT_W), device=device, dtype=torch.bfloat16
    )
    # Copy skip [1, CHANNELS, OUT_H, OUT_W] into out[:, :CHANNELS, :, :].
    # Both are contiguous, so we can view flat.
    first_half_flat = out.view(-1)[: CHANNELS * OUT_HW]
    skip_flat = skip.view(-1)
    n_copy = CHANNELS * OUT_HW

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_copy, COPY_BLOCK), 1, 1),
        _copy_first_half_kernel,
        (skip_flat, first_half_flat, n_copy, COPY_BLOCK),
    )

    # Second half write. Flatten out starting at offset CHANNELS*OHW.
    second_half_flat = out.view(-1)[CHANNELS * OUT_HW :]
    x_flat = x.view(-1)

    grid = (CHANNELS * OUT_H, ct.cdiv(OUT_W, BLOCK_W), 1)
    ct.launch(
        stream,
        grid,
        _resize_pad_second_half_kernel,
        (mean, x_flat, var, weight, bias, second_half_flat,
         CHANNELS, IN_H, IN_W, OUT_H, OUT_W_INTERP, OUT_W, IN_HW, OUT_HW, BLOCK_W),
    )
    return out
