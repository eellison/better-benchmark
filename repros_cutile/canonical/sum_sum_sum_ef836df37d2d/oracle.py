"""cuTile port of sum_sum_sum_ef836df37d2d: U-Net bilinear scatter + BN
backward.

Torch handles the bilinear scatter (via aten.index_put) and BN forward +
ReLU + where. cuTile handles BOTH the channel reductions (sum_1, sum_2,
sum_3) AND the BN-backward epilogue in-kernel, one program per channel —
mirroring Triton's `_bn_backward_kernel[(C,)]` structure.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.000423728813559322


@ct.kernel
def _bn_backward_kernel(
    producer_ptr,     # f32 [C, HW_PAD]
    centered_ptr,     # f32 [C, HW_PAD]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    sum1_ptr,         # f32 [C]
    mul13_ptr,        # f32 [C]
    sum4_ptr,         # f32 [C]
    out_ptr,          # bf16 [C, HW_PAD]
    HW_: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    c = ct.bid(0)
    producer = ct.load(producer_ptr, index=(c, 0), shape=(1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(centered_ptr, index=(c, 0), shape=(1, HW_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))

    # Build valid mask (col < HW).
    idx = ct.arange(HW_PAD, dtype=ct.int32)
    valid = ct.reshape(idx < HW_, (1, HW_PAD))
    zeros_f = ct.full((1, HW_PAD), 0.0, dtype=ct.float32)

    producer_m = ct.where(valid, producer, zeros_f)
    dot_prod = producer_m * centered
    dot_prod_m = ct.where(valid, dot_prod, zeros_f)

    sum_1 = ct.sum(producer_m)   # 0-d
    sum_2 = ct.sum(dot_prod_m)   # 0-d

    sum1_tile = ct.full((1,), sum_1, dtype=ct.float32)
    sum2_tile = ct.full((1,), sum_2, dtype=ct.float32)
    ct.store(sum1_ptr, index=(c,), tile=sum1_tile)
    ct.store(mul13_ptr, index=(c,), tile=sum2_tile * invstd)

    invstd_3d = ct.reshape(invstd, (1, 1))
    weight_3d = ct.reshape(weight, (1, 1))
    sum1_3d = ct.reshape(sum1_tile, (1, 1))
    sum2_3d = ct.reshape(sum2_tile, (1, 1))

    mean_term = sum1_3d * SCALE_C
    dot_scaled = sum2_3d * SCALE_C
    variance_term = dot_scaled * (invstd_3d * invstd_3d)
    output_scale = invstd_3d * weight_3d
    after_variance = producer - centered * variance_term
    after_mean = after_variance - mean_term
    out_f32 = after_mean * output_scale
    out_bf = ct.astype(out_f32, ct.bfloat16)
    ct.store(out_ptr, index=(c, 0), tile=out_bf)

    out_bf_f32 = ct.astype(out_bf, ct.float32)
    out_bf_m = ct.where(valid, out_bf_f32, zeros_f)
    sum_3 = ct.sum(out_bf_m)
    sum3_tile = ct.full((1,), sum_3, dtype=ct.float32)
    # Round through bf16 to match Triton's fp_downcast_rounding="rtne" then f32.
    sum3_bf = ct.astype(sum3_tile, ct.bfloat16)
    sum3_rounded = ct.astype(sum3_bf, ct.float32)
    ct.store(sum4_ptr, index=(c,), tile=sum3_rounded)


@oracle_impl(hardware="B200", point="bdfa8b76")
def oracle_forward(inputs):
    (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
     arg8, arg9, arg10, arg11, arg12, s0) = inputs
    device = arg0.device
    n = int(arg7.shape[0])
    c = int(arg7.shape[1])
    h = int(arg7.shape[2])
    w = int(arg7.shape[3])
    hw = h * w
    assert n == 1

    # Bilinear scatter (torch: mirrors Triton's _scatter_kernel + _combine_scatter_kernel).
    slice_1 = arg0[:, 512:1024]
    pad = torch.nn.functional.pad(slice_1, [0, -1, 0, 0])
    conv = pad.to(torch.float32)
    mul = conv * arg1
    add = conv - mul
    mul_1 = mul * arg2
    add_1 = mul - mul_1
    mul_2 = add * arg2
    add_2 = add - mul_2
    full_ = torch.zeros(tuple(int(x) for x in s0), dtype=torch.float32, device=device)
    ip_0 = torch.ops.aten.index_put.default(full_, [None, None, arg3, arg4], mul_1, True)
    ip_1 = torch.ops.aten.index_put.default(full_, [None, None, arg3, arg5], add_1, True)
    add_3 = ip_0 + ip_1
    ip_2 = torch.ops.aten.index_put.default(full_, [None, None, arg6, arg4], mul_2, True)
    add_4 = add_3 + ip_2
    ip_3 = torch.ops.aten.index_put.default(full_, [None, None, arg6, arg5], add_2, True)
    add_5 = add_4 + ip_3
    conv_1_bf = add_5.to(torch.bfloat16)

    # BN forward + ReLU + where (torch — no reductions here; matches producer arithmetic).
    sub = arg7.to(torch.float32) - arg8
    mul_3 = sub * arg9
    mul_4 = mul_3 * arg10.view(c, 1, 1)
    add_6 = mul_4 + arg11.view(c, 1, 1)
    bn_bf = add_6.to(torch.bfloat16)
    relu = torch.relu(bn_bf)
    le = relu <= 0
    where_result = torch.where(le, arg12, conv_1_bf)
    producer_f = where_result.to(torch.float32).contiguous()
    centered_f = sub.contiguous()

    squeeze_1 = arg9.view(c)
    hw_pad = 1 << (hw - 1).bit_length()

    # 2D [C, HW] views for the kernel (n == 1).
    producer_2d = producer_f.view(c, hw)
    centered_2d = centered_f.view(c, hw)

    out_2d = torch.empty((c, hw), device=device, dtype=torch.bfloat16)
    sum_1_out = torch.empty((c,), device=device, dtype=torch.float32)
    mul_13_out = torch.empty((c,), device=device, dtype=torch.float32)
    sum_3_out = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_backward_kernel,
        (producer_2d, centered_2d, squeeze_1, arg10,
         sum_1_out, mul_13_out, sum_3_out, out_2d,
         hw, hw_pad, SCALE),
    )
    out_contig = out_2d.view(n, c, h, w)
    return sum_1_out, mul_13_out, out_contig, sum_3_out
