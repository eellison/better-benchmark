"""cuTile port of sum_sum_sum_ee5e53038768: ConvNeXtV2 GRN/GELU multi-output.

Multi-output ConvNeXtV2 backward: computes out0, out1 (channel reductions),
out2 (scalar zero), out3 (dense bf16 [128, 2560, 7, 7]), out4 (bf16 channel sum).

The heavy dense output (out3) is produced by a cuTile kernel; torch handles
the reductions and simple scalars.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 2560
H = 7
W = 7
HW = H * W  # 49


@ct.kernel
def _dense_output_kernel(
    arg0_ptr,       # bf16 [N, C, HW]  (channels-last NHWC-strided; passed via contiguous view)
    arg1_ptr,       # bf16 [N, C, HW]
    weight_ptr,     # f32 [C]
    norm_ptr,       # f32 [N, C]
    denom_ptr,      # f32 [N]
    sum3_ptr,       # f32 [N, C]
    row_corr_ptr,   # f32 [N]
    zero_ptr,       # f32 [1]
    div4_num_ptr,   # bf16 [N, C, HW]  = to_bf16(gelu(arg1))  (precomputed)
    add2_denom_ptr, # f32 [N, C, HW]   (unused, kept for future)
    out_ptr,        # bf16 [N, C, HW]
    HW_C: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C, 1). Loops over HW blocks."""
    n = ct.bid(0)
    c = ct.bid(1)

    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    weight_s = ct.reshape(weight, ())
    norm = ct.astype(ct.load(norm_ptr, index=(n, c), shape=(1, 1)), ct.float32)
    norm_s = ct.reshape(norm, ())
    denom = ct.astype(ct.load(denom_ptr, index=(n,), shape=(1,)), ct.float32)
    denom_s = ct.reshape(denom, ())
    s3 = ct.astype(ct.load(sum3_ptr, index=(n, c), shape=(1, 1)), ct.float32)
    s3_s = ct.reshape(s3, ())
    row_corr = ct.astype(ct.load(row_corr_ptr, index=(n,), shape=(1,)), ct.float32)
    row_corr_s = ct.reshape(row_corr, ())
    zero_val = ct.astype(ct.load(zero_ptr, index=(0,), shape=(1,)), ct.float32)
    zero_s = ct.reshape(zero_val, ())

    # Precomputed values.
    div = norm_s / denom_s
    add2 = s3_s / denom_s + row_corr_s * 0.000390625
    is_zero = norm_s == 0.0

    for block in ct.static_iter(range(HW_PAD // BLOCK_HW)):
        hw_offsets = block * BLOCK_HW + ct.arange(BLOCK_HW, dtype=ct.int32)
        hw_mask = hw_offsets < HW_C

        a = ct.astype(
            ct.load(arg0_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        a_1d = ct.reshape(a, (BLOCK_HW,))
        gelu = ct.astype(
            ct.load(div4_num_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        gelu_1d = ct.reshape(gelu, (BLOCK_HW,))
        b = ct.astype(
            ct.load(arg1_ptr, index=(n, c, block), shape=(1, 1, BLOCK_HW),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        b_1d = ct.reshape(b, (BLOCK_HW,))

        # add1 = to_bf16(to_bf16(a) + to_bf16(a * w * div))
        add1_delta_bf = ct.astype(a_1d * weight_s * div, ct.bfloat16)
        add1_delta_f = ct.astype(add1_delta_bf, ct.float32)
        a_bf_f = ct.astype(ct.astype(a_1d, ct.bfloat16), ct.float32)
        add1 = ct.astype(a_bf_f + add1_delta_f, ct.bfloat16)
        add1_f = ct.astype(add1, ct.float32)

        # where = (div / gelu) if norm != 0 else zero
        where_val = ct.where(is_zero, zero_s, gelu_1d / norm_s)  # gelu/norm  (only where norm != 0)
        # add3_delta = to_bf16(add2 * where)
        add3_delta_bf = ct.astype(add2 * where_val, ct.bfloat16)
        add3_delta_f = ct.astype(add3_delta_bf, ct.float32)
        add3 = ct.astype(add1_f + add3_delta_f, ct.bfloat16)
        add3_f = ct.astype(ct.astype(add3, ct.bfloat16), ct.float32)  # keep for downstream

        # grad = 0.5*(erf(b/sqrt2)+1) + b * exp(-b^2/2) * 0.3989...
        # We need erf; approximate with Abramowitz-Stegun 7.1.26.
        # For NUMerical accuracy, compute erf outside kernel and pass through?
        # Approx: erf(x) ~ sign(x) * (1 - t * exp(-x^2 - poly(t)))
        # Use polynomial approx:
        # ax = |x|, t = 1/(1+0.3275911*ax)
        # erf ~ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2), sign = sign(x)
        # a1..a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        # For simplicity, avoid inner erf; compute add4 as (erf_term * 0.5 + b * exp(-b^2/2) * 0.3989)
        # where erf_term ~ erf(b/sqrt2) + 1.  Pass in precomputed erf_term (add_val) instead.
        # Simpler: precompute grad outside as f32 and pass in.
        # For now, use polynomial approx here.
        x = b_1d * 0.7071067811865476
        ax = ct.where(x < 0.0, -x, x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        erf_val = 1.0 - poly * ct.exp(-x * x)
        erf_signed = ct.where(x < 0.0, -erf_val, erf_val)
        add_val = erf_signed + 1.0

        grad = add_val * 0.5 + b_1d * ct.exp(-b_1d * b_1d * 0.5) * 0.3989422804014327
        result_bf = ct.astype(add3_f * grad, ct.bfloat16)

        # Only write valid positions.
        # Simplest way: pad out_ptr to HW_PAD, then narrow after. But we can't easily do that.
        # Since HW_PAD == HW (49 not power-of-2 — no, we require HW % BLOCK_HW == 0).
        # We'll require BLOCK_HW divides HW.
        ct.store(out_ptr, index=(n, c, block), tile=ct.reshape(result_bf, (1, 1, BLOCK_HW)))


@ct.kernel
def _channel_sum_kernel(
    dense_ptr,      # bf16 [N, C, HW]
    channel_out_ptr,# f32 [C]
    N_C: ct.Constant[int],
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    N_PAD: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C, 1, 1). Reduces over N*HW per channel block."""
    c_block = ct.bid(0)

    # Load (N_PAD, BLOCK_C, HW_PAD) tile.
    vals = ct.load(
        dense_ptr, index=(0, c_block, 0),
        shape=(N_PAD, BLOCK_C, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals_f = ct.astype(vals, ct.float32)
    # Wait: the reduction in the repro sums the bf16 tensor then converts to f32 — same result under summation.
    # ct.sum over (0, 2) axes; keep axis=1 (BLOCK_C).
    # ct.sum takes single axis or None. Do sum axis=0 then axis=1 (which becomes axis=1 of the result).
    partial = ct.sum(vals_f, axis=2)  # (N_PAD, BLOCK_C)
    total = ct.sum(partial, axis=0)   # (BLOCK_C,)
    ct.store(channel_out_ptr, index=(c_block,), tile=total)


@oracle_impl(hardware="B200", point="743aa381")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2 = inputs
    del _shape0, _shape1, _shape2
    device = arg0_1.device

    # Compute intermediates via torch (matches Repro exactly).
    with torch.no_grad():
        arg0_f = arg0_1.to(torch.float32)
        arg1_f = arg1_1.to(torch.float32)
        # mul_2 = arg1 * 0.5 * (erf(arg1/sqrt2) + 1)  == GELU(arg1)  (f32)
        erf_val = torch.special.erf(arg1_f * 0.7071067811865476)
        add_erf = erf_val + 1.0
        gelu_factor_f = 0.5 * add_erf  # for the dense kernel grad path (mul_12 = add * 0.5)
        mul_2_f = arg1_f * gelu_factor_f  # actual GELU value
        mul_2_bf16 = mul_2_f.to(torch.bfloat16)  # convert_element_type_3

        arg2_bc = arg2_1  # [N, C, 1, 1]
        arg3_bc = arg3_1  # [N, 1, 1, 1]
        div = arg2_bc / arg3_bc  # f32 [N, C, 1, 1]

        # mul_3 = to_bf16(mul_2) * div  (bf16 promoted to f32 * f32)
        mul_3_f = mul_2_bf16.to(torch.float32) * div  # f32 [N, C, H, W]
        mul_4_f = mul_3_f  # mul_scalar(1) identity
        mul_5 = arg0_f * mul_4_f
        sum_1 = arg0_f.sum(dim=(0, 2, 3), dtype=torch.float32)
        sum_2 = mul_5.sum(dim=(0, 2, 3), dtype=torch.float32)
        # mul_7 = arg0 * arg4 (broadcast); mul_8 = mul_7 * mul_2_bf16 (f32)
        mul_7 = arg0_f * arg4_1.view(1, C, 1, 1)
        mul_8 = mul_7 * mul_2_bf16.to(torch.float32)
        sum_3 = mul_8.sum(dim=(2, 3), keepdim=True, dtype=torch.float32)  # f32 [N, C, 1, 1]

        div_1 = div / arg3_bc  # [N, C, 1, 1]
        mul_10 = -sum_3 * div_1  # [N, C, 1, 1]
        row_corr = mul_10.sum(dim=1, dtype=torch.float32).squeeze(-1).squeeze(-1)  # f32 [N]

        out0 = sum_2
        out1 = sum_1
        out2 = torch.zeros((), device=device, dtype=torch.float32)

    # Dense output (out3) via cuTile.
    # Prepare NCHW-contiguous scratch tensors.
    arg0_nchw = arg0_1.contiguous().view(N, C, HW)
    arg1_nchw = arg1_1.contiguous().view(N, C, HW)
    # convert_element_type_3 = to_bf16(0.5 * (erf(arg1/sqrt2) + 1))
    gelu_factor_nchw = mul_2_bf16.contiguous().view(N, C, HW)
    weight_1d = arg4_1.contiguous().view(C)
    norm_2d = arg2_1.view(N, C).contiguous()
    denom_1d = arg3_1.view(N).contiguous()
    sum3_2d = sum_3.view(N, C).contiguous()
    row_corr_1d = row_corr.contiguous()
    zero_1 = out2.view(1)

    out3_nchw = torch.empty((N, C, HW), device=device, dtype=torch.bfloat16)
    # HW=49 → pad to 64 needed if using pow-2 block. But 49 doesn't divide any pow-2.
    # Use BLOCK_HW=1 (works but slow) OR pad HW dim.
    # Fastest workable: pad HW to 64 and mask writes via scatter.
    # Simplest: use BLOCK_HW=1 with a Python loop of 49 blocks per (n, c). 128*2560*49 = huge.
    # Reasonable: HW=49 requires per-position work. Use BLOCK_HW=49 with a pow-of-2 wrapper:
    # if we set HW_PAD=64 and BLOCK_HW=64 (one iteration per n,c), can we mask stores?
    # For a strided output where each (n, c, hw) is a unique element, store to (n, c, block=0)
    # with tile shape (1, 1, 64) would write 64 elements starting at hw=0, going 64 past.
    # That would go OOB (49 elements only).
    # Alternative: pad the output tensor to HW_PAD=64 elements per channel, then narrow.
    HW_PAD = 64
    BLOCK_HW = 64
    out3_padded = torch.empty((N, C, HW_PAD), device=device, dtype=torch.bfloat16)
    add2_denom_dummy = torch.empty((1,), device=device, dtype=torch.float32)  # unused

    # We also need padded input views for cuTile since load supports padding.
    # But input arrays with HW=49 will pad reads OK; write to HW_PAD tensor and narrow.

    # For arg0, arg1, gelu (padded reads), we use load with padding_mode=ZERO.
    # But arrays' logical HW=49. To use HW_PAD=64 we'd need to view the arrays as [N, C, HW_PAD]
    # with the last dim padded. Easiest: pad the inputs in torch first.
    def _pad_hw(t):
        return torch.nn.functional.pad(t.view(N, C, HW), (0, HW_PAD - HW))
    arg0_padded = _pad_hw(arg0_1).contiguous()
    arg1_padded = _pad_hw(arg1_1).contiguous()
    gelu_factor_padded = _pad_hw(mul_2_bf16).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C, 1),
        _dense_output_kernel,
        (
            arg0_padded, arg1_padded, weight_1d, norm_2d, denom_1d,
            sum3_2d, row_corr_1d, zero_1, gelu_factor_padded, add2_denom_dummy,
            out3_padded, HW, HW_PAD, BLOCK_HW,
        ),
    )

    # Narrow the padded output and copy to NHWC-strided out3.
    out3 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out3.copy_(out3_padded[:, :, :HW].view(N, C, H, W))

    # out4: bf16 channel sum of out3 -> f32.
    # torch approach (matches exactly):
    out4 = out3.sum(dim=(0, 2, 3), dtype=torch.bfloat16).to(torch.float32)

    return out0, out1, out2, out3, out4
