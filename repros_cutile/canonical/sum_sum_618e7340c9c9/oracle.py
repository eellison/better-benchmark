"""cuTile port of sum_sum_618e7340c9c9: DenseNet BN-backward + residual slice add.

The Triton oracle uses inline `add.rn.f32`, `sub.rn.f32`, `mul.rn.f32` PTX
plus explicit bf16 rounding boundaries — cuTile is RTNE by default so those
become plain `+`, `-`, `*`, and `ct.astype(x, ct.bfloat16)`.

The layout differs slightly between the two shape points (H,W = 14 or 7).
Torch handles the reductions and the residual slice adds; cuTile handles
the elementwise BN-backward epilogue kernel that produces the dense f32/bf16
outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.0012755102040816326  # 1/(4*14*14) = 1/784
C = 640
SLICE_START = 608
SLICE_C = 32  # 640 - 608


@ct.kernel
def _bn_backward_kernel(
    where_ptr,        # bf16 [N, C, HW]  (from torch.where(mask<=0, fill, source))
    centered_ptr,     # f32  [N, C, HW]  (arg15 - mean broadcast)
    mean_term_ptr,    # f32  [C]
    variance_ptr,     # f32  [C]
    output_scale_ptr, # f32  [C]
    out_bf16_ptr,     # bf16 [N, C, HW]
    HW_PAD: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    where_bf16 = ct.load(
        where_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = ct.load(
        centered_ptr, index=(n, c, 0), shape=(1, 1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean_term = ct.load(mean_term_ptr, index=(c,), shape=(1,))
    variance_term = ct.load(variance_ptr, index=(c,), shape=(1,))
    output_scale = ct.load(output_scale_ptr, index=(c,), shape=(1,))

    where_f = ct.astype(where_bf16, ct.float32)
    mean_3d = ct.reshape(mean_term, (1, 1, 1))
    var_3d = ct.reshape(variance_term, (1, 1, 1))
    scale_3d = ct.reshape(output_scale, (1, 1, 1))

    after_variance = where_f - centered * var_3d
    after_mean = after_variance - mean_3d
    dense_f32 = after_mean * scale_3d
    dense_bf16 = ct.astype(dense_f32, ct.bfloat16)
    ct.store(out_bf16_ptr, index=(n, c, 0), tile=dense_bf16)


def _launch(inputs, *, H: int, W: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1,
        arg12_1, arg13_1, arg14_1, arg15_1,
        arg16_1, arg17_1, arg18_1,
    ) = inputs
    device = arg0_1.device
    N = 4  # from shapes
    hw = H * W

    # Residual add: sum of arg0..arg11 sliced [608:640]
    # matching the eager exactly with bf16 accumulation semantics.
    # We rely on torch's bf16 add matching Inductor's expectation.
    resid = arg0_1[:, SLICE_START:C].contiguous()
    for a in (arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
              arg7_1, arg8_1, arg9_1, arg10_1, arg11_1):
        resid = resid + a[:, SLICE_START:C]

    # where(arg12 <= 0, arg13, arg14) — bf16
    le = arg12_1 <= 0
    where_result = torch.where(le, arg13_1, arg14_1)  # bf16 (N, C, H, W)
    where_bf16 = where_result.contiguous()

    # Reductions
    where_f = where_bf16.to(torch.float32)
    sum_1 = where_f.sum(dim=(0, 2, 3))  # (C,) f32

    # centered = arg15 (bf16) as f32 - mean broadcast
    arg15_f = arg15_1.to(torch.float32)
    centered = arg15_f - arg16_1  # (N, C, H, W) f32
    sum_2 = (where_f * centered).sum(dim=(0, 2, 3))  # (C,)

    # Coefficients:
    mean_term = sum_1 * SCALE
    prod_scaled = sum_2 * SCALE
    invstd_sq = arg17_1 * arg17_1
    variance_term = prod_scaled * invstd_sq  # (C,)
    output_scale = arg17_1 * arg18_1  # (C,)
    mul_8 = sum_2 * arg17_1  # (C,) — scale_grad output

    # Cutile kernel: BN-backward dense
    dense_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * hw, hw, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    hw_pad = 1 << (hw - 1).bit_length()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, C, 1), _bn_backward_kernel,
        (
            where_bf16.view(N, C, hw),
            centered.contiguous().view(N, C, hw),
            mean_term, variance_term, output_scale,
            dense_bf16.view(N, C, hw),
            hw_pad,
        ),
    )

    # Final add: residual + dense_bf16[:, 608:640]
    add_out = resid + dense_bf16[:, SLICE_START:C]
    return sum_1, mul_8, dense_bf16, add_out


@oracle_impl(hardware="B200", point="94c34cc8", H=14, W=14)
@oracle_impl(hardware="B200", point="157fe920", H=7, W=7)
def oracle_forward(inputs, *, H: int, W: int):
    return _launch(inputs, H=H, W=W)
