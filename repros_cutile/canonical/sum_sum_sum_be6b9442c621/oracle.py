"""cuTile port of sum_sum_sum_be6b9442c621: PyTorch U-Net bilinear-upsample-backward.

Multi-kernel:
  1. Bilinear scatter-add kernel: for each (c, spatial), compute the 4
     bilinear coefficients and atomic-add each into a separate scatter buffer
     at (row0/row1, col0/col1) targets.
  2. Combine scatters + downstream BN-backward: reuse the same combined
     scatter to compute the final bf16 output plus sum reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 128
SRC_H = 320
SRC_W = 478
SRC_STRIDE_H = 479
OUT_H = 160
OUT_W = 239
OUT_HW = OUT_H * OUT_W
SCALE = 2.615062761506276e-05


@ct.kernel
def _bilinear_scatter_kernel(
    src_ptr,          # bf16 [1, 256, 320, 479]
    scale_h_ptr,      # f32  [320]
    scale_w_ptr,      # f32  [478]
    row0_ptr,         # i64  [320]
    col0_ptr,         # i64  [478]
    col1_ptr,         # i64  [478]
    row1_ptr,         # i64  [320]
    scatter0_ptr,     # f32  [C, OUT_H, OUT_W]
    scatter1_ptr,     # f32  [C, OUT_H, OUT_W]
    scatter2_ptr,     # f32  [C, OUT_H, OUT_W]
    scatter3_ptr,     # f32  [C, OUT_H, OUT_W]
    C_: ct.Constant[int],
    SRC_HW: ct.Constant[int],
    SRC_H_: ct.Constant[int],
    SRC_W_: ct.Constant[int],
    OUT_H_: ct.Constant[int],
    OUT_W_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    k_block = ct.bid(1)
    off = ct.arange(BLOCK_K, dtype=ct.int32) + k_block * BLOCK_K
    active = off < SRC_HW

    h = off // SRC_W_
    w = off - h * SRC_W_

    c_src = ct.full((BLOCK_K,), c + C_, dtype=ct.int32)  # channels [128, 256)
    zero_b = ct.full((BLOCK_K,), 0, dtype=ct.int32)

    # Gather source (channels [128, 256))
    x = ct.astype(
        ct.gather(src_ptr, (zero_b, c_src, h, w), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    scale_h = ct.gather(scale_h_ptr, (h,), mask=active, padding_value=0.0)
    scale_w = ct.gather(scale_w_ptr, (w,), mask=active, padding_value=0.0)

    # Bilinear decomposition
    mul_ = x * scale_h
    add_ = x - mul_
    mul_1 = mul_ * scale_w
    add_1 = mul_ - mul_1
    mul_2 = add_ * scale_w
    add_2 = add_ - mul_2

    row0 = ct.astype(ct.gather(row0_ptr, (h,), mask=active, padding_value=0), ct.int32)
    col0 = ct.astype(ct.gather(col0_ptr, (w,), mask=active, padding_value=0), ct.int32)
    col1 = ct.astype(ct.gather(col1_ptr, (w,), mask=active, padding_value=0), ct.int32)
    row1 = ct.astype(ct.gather(row1_ptr, (h,), mask=active, padding_value=0), ct.int32)

    c_t = ct.full((BLOCK_K,), c, dtype=ct.int32)
    # Mask-out invalid rows by using out-of-bounds row index
    invalid_row = ct.full((BLOCK_K,), OUT_H_, dtype=ct.int32)
    row0_safe = ct.where(active, row0, invalid_row)
    row1_safe = ct.where(active, row1, invalid_row)

    ct.atomic_add(scatter0_ptr, (c_t, row0_safe, col0), mul_1)
    ct.atomic_add(scatter1_ptr, (c_t, row0_safe, col1), add_1)
    ct.atomic_add(scatter2_ptr, (c_t, row1_safe, col0), mul_2)
    ct.atomic_add(scatter3_ptr, (c_t, row1_safe, col1), add_2)


@ct.kernel
def _bn_reduce_kernel(
    scatter_bf_ptr,   # bf16 [C, OUT_H*OUT_W]
    x7_ptr,           # bf16 [1, 128, OUT_H, OUT_W]
    mean_ptr,         # f32  [128]
    invstd_ptr,       # f32  [128]
    weight_ptr,       # f32  [128]
    bias_ptr,         # f32  [128]
    scalar_ptr,       # bf16 [1]
    out_sum1_ptr,     # f32  [C]  atomic
    sum2_ptr,         # f32  [C]  atomic
    HW_: ct.Constant[int],
    OUT_W_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    hw_off = ct.arange(BLOCK_HW, dtype=ct.int32) + tile * BLOCK_HW
    active = hw_off < HW_

    h = hw_off // OUT_W_
    w = hw_off - h * OUT_W_

    zero_b = ct.full((BLOCK_HW,), 0, dtype=ct.int32)
    c_t = ct.full((BLOCK_HW,), c, dtype=ct.int32)

    x7 = ct.astype(
        ct.gather(x7_ptr, (zero_b, c_t, h, w), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))

    scatter = ct.astype(
        ct.gather(scatter_bf_ptr, (c_t, hw_off), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )

    centered = x7 - ct.broadcast_to(ct.astype(mean, ct.float32), (BLOCK_HW,))
    gate_mul = centered * ct.astype(invstd, ct.float32) * ct.astype(weight, ct.float32)
    gate = ct.astype(
        ct.astype(gate_mul + ct.astype(bias, ct.float32), ct.bfloat16),
        ct.float32,
    )
    active_gate = gate <= 0.0
    scalar_f = ct.astype(ct.astype(scalar, ct.bfloat16), ct.float32)
    y = ct.where(active_gate,
                 ct.broadcast_to(scalar_f, (BLOCK_HW,)),
                 scatter)

    sum1 = ct.where(active, y, 0.0)
    sum2 = ct.where(active, y * centered, 0.0)
    sum1_val = ct.sum(sum1)
    sum2_val = ct.sum(sum2)

    ct.atomic_add(out_sum1_ptr, c, sum1_val)
    ct.atomic_add(sum2_ptr, c, sum2_val)


@ct.kernel
def _bn_output_kernel(
    scatter_bf_ptr,
    x7_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    sum1_ptr,           # per-channel (final)
    sum2_ptr,           # per-channel (final)
    out_ptr,            # bf16 [1, C, OUT_H, OUT_W]
    out_sum4_ptr,       # f32  [C] atomic
    HW_: ct.Constant[int],
    OUT_W_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    hw_off = ct.arange(BLOCK_HW, dtype=ct.int32) + tile * BLOCK_HW
    active = hw_off < HW_
    h = hw_off // OUT_W_
    w = hw_off - h * OUT_W_

    zero_b = ct.full((BLOCK_HW,), 0, dtype=ct.int32)
    c_t = ct.full((BLOCK_HW,), c, dtype=ct.int32)

    x7 = ct.astype(
        ct.gather(x7_ptr, (zero_b, c_t, h, w), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    sum1 = ct.load(sum1_ptr, index=(c,), shape=(1,))
    sum2 = ct.load(sum2_ptr, index=(c,), shape=(1,))

    scatter = ct.astype(
        ct.gather(scatter_bf_ptr, (c_t, hw_off), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )

    mean_f = ct.astype(mean, ct.float32)
    invstd_f = ct.astype(invstd, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    sum1_f = ct.astype(sum1, ct.float32)
    sum2_f = ct.astype(sum2, ct.float32)

    centered = x7 - ct.broadcast_to(mean_f, (BLOCK_HW,))
    gate_mul = centered * invstd_f * weight_f
    gate_bf = ct.astype(gate_mul + ct.astype(bias, ct.float32), ct.bfloat16)
    gate = ct.astype(gate_bf, ct.float32)
    active_gate = gate <= 0.0
    scalar_f = ct.astype(ct.astype(scalar, ct.bfloat16), ct.float32)
    y = ct.where(active_gate,
                 ct.broadcast_to(scalar_f, (BLOCK_HW,)),
                 scatter)

    mean_term = sum1_f * SCALE_
    scaled_sum2 = sum2_f * SCALE_
    invstd_sq = invstd_f * invstd_f
    var_term = scaled_sum2 * invstd_sq
    affine_scale = invstd_f * weight_f

    correction = centered * ct.broadcast_to(var_term, (BLOCK_HW,))
    after = y - correction - ct.broadcast_to(mean_term, (BLOCK_HW,))
    out_f = after * ct.broadcast_to(affine_scale, (BLOCK_HW,))
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.scatter(out_ptr, (zero_b, c_t, h, w), out_bf, mask=active)

    out_bf_f = ct.astype(out_bf, ct.float32)
    out_bf_masked = ct.where(active, out_bf_f, 0.0)
    sum4 = ct.sum(out_bf_masked)
    ct.atomic_add(out_sum4_ptr, c, sum4)


@oracle_impl(hardware="B200", point="0ba1fdea", SRC_BLOCK=256, HW_BLOCK=512,
             ZERO_BLOCK=1024, scatter_warps=8, final_warps=8)
def oracle_forward(inputs, *, SRC_BLOCK, HW_BLOCK, ZERO_BLOCK, scatter_warps, final_warps):
    del ZERO_BLOCK, scatter_warps, final_warps
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
        arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1,
        _shape_param_0,
    ) = inputs
    device = arg0_1.device

    scatter0 = torch.zeros((C, OUT_H, OUT_W), device=device, dtype=torch.float32)
    scatter1 = torch.zeros((C, OUT_H, OUT_W), device=device, dtype=torch.float32)
    scatter2 = torch.zeros((C, OUT_H, OUT_W), device=device, dtype=torch.float32)
    scatter3 = torch.zeros((C, OUT_H, OUT_W), device=device, dtype=torch.float32)

    # Prepare 1D views of the indices
    scale_h_1d = arg1_1.view(SRC_H)     # [320]
    scale_w_1d = arg2_1.view(SRC_W)     # [478]
    row0_1d = arg3_1.view(SRC_H)
    col0_1d = arg4_1.view(SRC_W)
    col1_1d = arg5_1.view(SRC_W)
    row1_1d = arg6_1.view(SRC_H)

    src_hw = SRC_H * SRC_W
    src_tiles = (src_hw + SRC_BLOCK - 1) // SRC_BLOCK

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, src_tiles, 1), _bilinear_scatter_kernel,
        (arg0_1, scale_h_1d, scale_w_1d, row0_1d, col0_1d, col1_1d, row1_1d,
         scatter0, scatter1, scatter2, scatter3,
         C, src_hw, SRC_H, SRC_W, OUT_H, OUT_W, SRC_BLOCK),
    )

    combined = scatter0 + scatter1 + scatter2 + scatter3
    scatter_bf = combined.to(torch.bfloat16).view(C, OUT_HW)

    mean_1d = arg8_1.view(C)
    invstd_1d = arg9_1.view(C)
    weight_1d = arg10_1
    bias_1d = arg11_1
    scalar_1d = arg12_1.view(1)

    out_sum1 = torch.zeros((C,), device=device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=device, dtype=torch.float32)

    hw_tiles = (OUT_HW + HW_BLOCK - 1) // HW_BLOCK
    ct.launch(
        stream, (C, hw_tiles, 1), _bn_reduce_kernel,
        (scatter_bf, arg7_1, mean_1d, invstd_1d, weight_1d, bias_1d,
         scalar_1d, out_sum1, sum2,
         OUT_HW, OUT_W, HW_BLOCK),
    )

    out_mul13 = sum2 * invstd_1d

    out_tensor = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_sum4_acc = torch.zeros((C,), device=device, dtype=torch.float32)

    ct.launch(
        stream, (C, hw_tiles, 1), _bn_output_kernel,
        (scatter_bf, arg7_1, mean_1d, invstd_1d, weight_1d, bias_1d,
         scalar_1d, out_sum1, sum2, out_tensor, out_sum4_acc,
         OUT_HW, OUT_W, SCALE, HW_BLOCK),
    )
    out_sum4 = out_sum4_acc.to(torch.bfloat16).to(torch.float32)

    return out_sum1, out_mul13, out_tensor, out_sum4
