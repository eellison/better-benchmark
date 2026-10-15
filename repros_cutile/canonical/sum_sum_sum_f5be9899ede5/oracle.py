"""cuTile port of sum_sum_sum_f5be9899ede5: U-Net bilinear upsample backward.

Torch handles the bilinear-upsample-backward scatter chain (index_put with
accumulate=True) plus BN forward+ReLU+where producer. cuTile handles the
BN-backward channel reductions AND the final bf16 gradient output, using
a per-tile partial + per-channel finalize structure mirroring Triton's
`_sum12_partial_kernel` + `_sum12_finalize_kernel` + `_final_output_partial_kernel`
+ `_sum4_finalize_kernel`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 6.524008350730689e-06
C = 64
OUT_H = 320
OUT_W = 479
OUT_HW = OUT_H * OUT_W  # 153280
HW_BLOCK = 512
HW_TILES = (OUT_HW + HW_BLOCK - 1) // HW_BLOCK  # 300 (ceil)


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


HW_TILES_PAD = _next_pow2(HW_TILES)  # 512


@ct.kernel
def _sum12_partial_kernel(
    producer_ptr,     # f32 [C, OUT_HW]
    centered_ptr,     # f32 [C, OUT_HW]
    partial_sum1_ptr, # f32 [C, HW_TILES]
    partial_sum2_ptr, # f32 [C, HW_TILES]
    HW_: ct.Constant[int],
    HW_BLOCK_: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    producer = ct.load(producer_ptr, index=(c, tile), shape=(1, HW_BLOCK_),
                       padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(centered_ptr, index=(c, tile), shape=(1, HW_BLOCK_),
                       padding_mode=ct.PaddingMode.ZERO)

    hw_idx = tile * HW_BLOCK_ + ct.arange(HW_BLOCK_, dtype=ct.int32)
    valid = ct.reshape(hw_idx < HW_, (1, HW_BLOCK_))
    zeros = ct.full((1, HW_BLOCK_), 0.0, dtype=ct.float32)
    producer_m = ct.where(valid, producer, zeros)
    dot_m = ct.where(valid, producer_m * centered, zeros)

    s1 = ct.sum(producer_m)
    s2 = ct.sum(dot_m)
    ct.store(partial_sum1_ptr, index=(c, tile), tile=ct.reshape(ct.full((1,), s1, dtype=ct.float32), (1, 1)))
    ct.store(partial_sum2_ptr, index=(c, tile), tile=ct.reshape(ct.full((1,), s2, dtype=ct.float32), (1, 1)))


@ct.kernel
def _sum12_finalize_kernel(
    partial_sum1_ptr, # f32 [C, HW_TILES]
    partial_sum2_ptr, # f32 [C, HW_TILES]
    invstd_ptr,       # f32 [C]
    sum1_ptr,         # f32 [C]
    sum2_ptr,         # f32 [C]
    mul13_ptr,        # f32 [C]
    TILES_: ct.Constant[int],
    TILES_PAD: ct.Constant[int],
):
    c = ct.bid(0)
    p1 = ct.load(partial_sum1_ptr, index=(c, 0), shape=(1, TILES_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    p2 = ct.load(partial_sum2_ptr, index=(c, 0), shape=(1, TILES_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    idx = ct.arange(TILES_PAD, dtype=ct.int32)
    valid = ct.reshape(idx < TILES_, (1, TILES_PAD))
    zeros = ct.full((1, TILES_PAD), 0.0, dtype=ct.float32)
    p1_m = ct.where(valid, p1, zeros)
    p2_m = ct.where(valid, p2, zeros)
    sum1 = ct.sum(p1_m)
    sum2 = ct.sum(p2_m)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    sum1_tile = ct.full((1,), sum1, dtype=ct.float32)
    sum2_tile = ct.full((1,), sum2, dtype=ct.float32)
    ct.store(sum1_ptr, index=(c,), tile=sum1_tile)
    ct.store(sum2_ptr, index=(c,), tile=sum2_tile)
    ct.store(mul13_ptr, index=(c,), tile=sum2_tile * invstd)


@ct.kernel
def _final_output_partial_kernel(
    producer_ptr,     # f32 [C, OUT_HW]
    centered_ptr,     # f32 [C, OUT_HW]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    sum1_ptr,         # f32 [C]
    sum2_ptr,         # f32 [C]
    out_ptr,          # bf16 [C, OUT_HW]
    partial_sum4_ptr, # f32 [C, HW_TILES]
    HW_: ct.Constant[int],
    HW_BLOCK_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    producer = ct.load(producer_ptr, index=(c, tile), shape=(1, HW_BLOCK_),
                       padding_mode=ct.PaddingMode.ZERO)
    centered = ct.load(centered_ptr, index=(c, tile), shape=(1, HW_BLOCK_),
                       padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum1 = ct.load(sum1_ptr, index=(c,), shape=(1,))
    sum2 = ct.load(sum2_ptr, index=(c,), shape=(1,))

    hw_idx = tile * HW_BLOCK_ + ct.arange(HW_BLOCK_, dtype=ct.int32)
    valid = ct.reshape(hw_idx < HW_, (1, HW_BLOCK_))
    zeros = ct.full((1, HW_BLOCK_), 0.0, dtype=ct.float32)

    invstd_2d = ct.reshape(invstd, (1, 1))
    weight_2d = ct.reshape(weight, (1, 1))
    sum1_2d = ct.reshape(sum1, (1, 1))
    sum2_2d = ct.reshape(sum2, (1, 1))

    mean_term = sum1_2d * SCALE_
    dot_scaled = sum2_2d * SCALE_
    var_term = dot_scaled * (invstd_2d * invstd_2d)
    output_scale = invstd_2d * weight_2d

    after_var = producer - centered * var_term
    after_mean = after_var - mean_term
    out_f = after_mean * output_scale
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(c, tile), tile=out_bf)

    out_bf_f = ct.astype(out_bf, ct.float32)
    out_m = ct.where(valid, out_bf_f, zeros)
    p4 = ct.sum(out_m)
    ct.store(partial_sum4_ptr, index=(c, tile),
             tile=ct.reshape(ct.full((1,), p4, dtype=ct.float32), (1, 1)))


@ct.kernel
def _sum4_finalize_kernel(
    partial_sum4_ptr, # f32 [C, HW_TILES]
    out_sum4_ptr,     # f32 [C]
    TILES_: ct.Constant[int],
    TILES_PAD: ct.Constant[int],
):
    c = ct.bid(0)
    p4 = ct.load(partial_sum4_ptr, index=(c, 0), shape=(1, TILES_PAD),
                 padding_mode=ct.PaddingMode.ZERO)
    idx = ct.arange(TILES_PAD, dtype=ct.int32)
    valid = ct.reshape(idx < TILES_, (1, TILES_PAD))
    zeros = ct.full((1, TILES_PAD), 0.0, dtype=ct.float32)
    p4_m = ct.where(valid, p4, zeros)
    total = ct.sum(p4_m)
    tot_tile = ct.full((1,), total, dtype=ct.float32)
    # Round through bf16 (matches Triton's fp_downcast_rounding="rtne" then f32).
    tot_bf = ct.astype(tot_tile, ct.bfloat16)
    tot_rounded = ct.astype(tot_bf, ct.float32)
    ct.store(out_sum4_ptr, index=(c,), tile=tot_rounded)


@oracle_impl(hardware="B200", point="b518aad7")
def oracle_forward(inputs):
    (
        arg0_1,  # bf16[1,128,640,959]
        arg1_1,  # f32[640,1]
        arg2_1,  # f32[958]
        arg3_1,  # i64[640,1]
        arg4_1,  # i64[958]
        arg5_1,  # i64[958]
        arg6_1,  # i64[640,1]
        arg7_1,  # bf16[1,64,320,479]
        arg8_1,  # f32[1,64,1,1]
        arg9_1,  # f32[1,64,1,1]
        arg10_1, # f32[64]
        arg11_1, # f32[64]
        arg12_1, # bf16scalar
        _shape_param_0,
    ) = inputs
    device = arg0_1.device

    slice_1 = arg0_1[:, 64:128]
    constant_pad_nd = torch.nn.functional.pad(slice_1, [0, -1, 0, 0])
    conv0 = constant_pad_nd.to(torch.float32)
    mul = conv0 * arg1_1
    neg = -mul
    add = conv0 + neg
    mul_1 = mul * arg2_1
    neg_1 = -mul_1
    add_1 = mul + neg_1
    mul_2 = add * arg2_1
    neg_2 = -mul_2
    add_2 = add + neg_2

    full = torch.zeros((1, 64, 320, 479), dtype=torch.float32, device=device)
    index_put = torch.ops.aten.index_put.default(
        full, [None, None, arg3_1, arg4_1], mul_1, True
    )
    index_put_1 = torch.ops.aten.index_put.default(
        full, [None, None, arg3_1, arg5_1], add_1, True
    )
    add_3 = index_put + index_put_1
    index_put_2 = torch.ops.aten.index_put.default(
        full, [None, None, arg6_1, arg4_1], mul_2, True
    )
    add_4 = add_3 + index_put_2
    index_put_3 = torch.ops.aten.index_put.default(
        full, [None, None, arg6_1, arg5_1], add_2, True
    )
    add_5 = add_4 + index_put_3
    conv1 = add_5.to(torch.bfloat16)

    # BN forward + ReLU + where (torch producer, matches Triton's producer path).
    sub = arg7_1 - arg8_1
    mul_3 = sub * arg9_1
    unsq1 = arg10_1.unsqueeze(-1).unsqueeze(-1)
    mul_4 = mul_3 * unsq1
    unsq3 = arg11_1.unsqueeze(-1).unsqueeze(-1)
    add_6 = mul_4 + unsq3
    conv2 = add_6.to(torch.bfloat16)
    relu = torch.relu(conv2)
    le = relu <= 0
    where_out = torch.where(le, arg12_1, conv1)
    producer_f = where_out.to(torch.float32).contiguous()  # [1, 64, 320, 479]
    centered_f = sub.contiguous()  # [1, 64, 320, 479]

    invstd_1d = arg9_1.view(C).contiguous()
    weight_1d = arg10_1.view(C).contiguous()
    producer_2d = producer_f.view(C, OUT_HW)
    centered_2d = centered_f.view(C, OUT_HW)

    stream = torch.cuda.current_stream()

    # Kernel 1: partial sum_1, sum_2 across HW tiles.
    partial_sum1 = torch.empty((C, HW_TILES), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, HW_TILES), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, HW_TILES, 1),
        _sum12_partial_kernel,
        (producer_2d, centered_2d, partial_sum1, partial_sum2, OUT_HW, HW_BLOCK),
    )

    # Kernel 2: finalize sum_1, sum_2, mul_13.
    sum1_out = torch.empty((C,), device=device, dtype=torch.float32)
    sum2_out = torch.empty((C,), device=device, dtype=torch.float32)
    mul13_out = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, 1, 1),
        _sum12_finalize_kernel,
        (partial_sum1, partial_sum2, invstd_1d,
         sum1_out, sum2_out, mul13_out,
         HW_TILES, HW_TILES_PAD),
    )

    # Kernel 3: compute output + partial sum_4.
    dense_out = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_out_2d = dense_out.view(C, OUT_HW)
    partial_sum4 = torch.empty((C, HW_TILES), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, HW_TILES, 1),
        _final_output_partial_kernel,
        (producer_2d, centered_2d, invstd_1d, weight_1d,
         sum1_out, sum2_out, dense_out_2d, partial_sum4,
         OUT_HW, HW_BLOCK, REDUCE_SCALE),
    )

    # Kernel 4: finalize sum_4 (bf16 rounded).
    sum4_out = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (C, 1, 1),
        _sum4_finalize_kernel,
        (partial_sum4, sum4_out, HW_TILES, HW_TILES_PAD),
    )

    return sum1_out, mul13_out, dense_out, sum4_out
