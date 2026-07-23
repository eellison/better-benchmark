"""cuTile port of sum_sum_sum_12f47639e40c: UNet upsample_bwd + BN backward.

Do the index_put chain on torch side (complex layout), do the fused pointwise
BN-backward + relu-gate step in a cuTile kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fused_kernel(
    where_ptr, arg7_ptr, mean_ptr, mul6_ptr, weight_ptr, scale_ptr,
    out_bwd_ptr,
    C: ct.Constant[int], SPATIAL: ct.Constant[int],
    BLOCK_C: ct.Constant[int], BLOCK_S: ct.Constant[int],
):
    c_block = ct.bid(0)
    s_block = ct.bid(1)

    where_val = ct.load(where_ptr, index=(c_block, s_block), shape=(BLOCK_C, BLOCK_S))
    arg7 = ct.load(arg7_ptr, index=(c_block, s_block), shape=(BLOCK_C, BLOCK_S))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    mul6 = ct.load(mul6_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    scale = ct.load(scale_ptr, index=(c_block,), shape=(BLOCK_C,))

    where_f = ct.astype(where_val, ct.float32)
    arg7_f = ct.astype(arg7, ct.float32)
    mean_2d = ct.reshape(mean, (BLOCK_C, 1))
    mul6_2d = ct.reshape(mul6, (BLOCK_C, 1))
    weight_2d = ct.reshape(weight, (BLOCK_C, 1))
    scale_2d = ct.reshape(scale, (BLOCK_C, 1))

    sub_1 = arg7_f - mean_2d
    mul_11 = sub_1 * weight_2d
    sub_2 = where_f - mul_11
    sub_3 = sub_2 - mul6_2d
    mul_12 = sub_3 * scale_2d
    out_bf = ct.astype(mul_12, ct.bfloat16)
    ct.store(out_bwd_ptr, index=(c_block, s_block), tile=out_bf)


@ct.kernel
def _sum3_kernel(
    out_bwd_ptr,      # bf16 [C, PADDED_S]
    out_sum3_ptr,     # f32 [C, num_s_blocks]
    SPATIAL: ct.Constant[int],
    BLOCK_C: ct.Constant[int], BLOCK_S: ct.Constant[int],
):
    c_block = ct.bid(0)
    s_block = ct.bid(1)
    x_bf = ct.load(out_bwd_ptr, index=(c_block, s_block), shape=(BLOCK_C, BLOCK_S))
    s_off = s_block * BLOCK_S
    cols = ct.arange(BLOCK_S, dtype=ct.int32)
    col_mask = (s_off + cols) < SPATIAL
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_S))
    zero = ct.full((BLOCK_C, BLOCK_S), 0.0, dtype=ct.float32)
    x_f = ct.where(col_mask_2d, ct.astype(x_bf, ct.float32), zero)
    part = ct.sum(x_f, axis=1, keepdims=True)
    ct.store(out_sum3_ptr, index=(c_block, s_block), tile=part)


@ct.kernel
def _channel_sum_bf16_kernel(
    partials_ptr,     # f32 [C, num_blocks]
    out_ptr,          # f32 [C]
    NUM_BLOCKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    partials = ct.load(partials_ptr, index=(c_block, 0), shape=(BLOCK_C, NUM_BLOCKS))
    s = ct.sum(partials, axis=1)  # (BLOCK_C,)
    s_bf = ct.astype(ct.astype(s, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(c_block,), tile=s_bf)


@ct.kernel
def _sum12_kernel(
    where_ptr, arg7_ptr, mean_ptr,
    out_sum1_ptr, out_sum2_ptr,
    C: ct.Constant[int], SPATIAL: ct.Constant[int],
    BLOCK_C: ct.Constant[int], BLOCK_S: ct.Constant[int],
):
    c_block = ct.bid(0)
    s_block = ct.bid(1)
    where_val = ct.load(where_ptr, index=(c_block, s_block), shape=(BLOCK_C, BLOCK_S))
    arg7 = ct.load(arg7_ptr, index=(c_block, s_block), shape=(BLOCK_C, BLOCK_S))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean, (BLOCK_C, 1))

    y = ct.astype(where_val, ct.float32)
    centered = ct.astype(arg7, ct.float32) - mean_2d
    # Mask out contribution from padded columns beyond SPATIAL.
    s_off = s_block * BLOCK_S
    cols = ct.arange(BLOCK_S, dtype=ct.int32)
    col_mask = (s_off + cols) < SPATIAL
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_S))
    zero = ct.full((BLOCK_C, BLOCK_S), 0.0, dtype=ct.float32)
    y_m = ct.where(col_mask_2d, y, zero)
    yc_m = ct.where(col_mask_2d, y * centered, zero)
    part1 = ct.sum(y_m, axis=1, keepdims=True)  # (BLOCK_C, 1)
    part2 = ct.sum(yc_m, axis=1, keepdims=True)  # (BLOCK_C, 1)
    ct.store(out_sum1_ptr, index=(c_block, s_block), tile=part1)
    ct.store(out_sum2_ptr, index=(c_block, s_block), tile=part2)


@ct.kernel
def _channel_sum_kernel(
    partials_ptr,     # f32 [C, num_blocks]
    out_ptr,          # f32 [C]
    NUM_BLOCKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    partials = ct.load(partials_ptr, index=(c_block, 0), shape=(BLOCK_C, NUM_BLOCKS))
    s = ct.sum(partials, axis=1)
    ct.store(out_ptr, index=(c_block,), tile=s)


@oracle_impl(hardware="B200", point="89c70199")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, _shape) = inputs
    device = arg0_1.device

    # Do the index_put chain torch-side (bilinear-upsample backward).
    C_full = arg0_1.shape[1]  # 512
    slice_1 = arg0_1[:, C_full // 2:, :, :]  # (1, 256, 160, 239)
    pad = torch.nn.functional.pad(slice_1, [0, -1, 0, 0])  # (1, 256, 160, 238)
    conv = pad.to(torch.float32)
    mul = conv * arg1_1
    neg = -mul
    add = conv + neg
    mul_1 = mul * arg2_1
    neg_1 = -mul_1
    add_1 = mul + neg_1
    mul_2 = add * arg2_1
    neg_2 = -mul_2
    add_2 = add + neg_2

    full = torch.zeros((1, 256, 80, 119), device=device, dtype=torch.float32)
    ip_result = torch.ops.aten.index_put.default(full, [None, None, arg3_1, arg4_1], mul_1, True)
    ip2 = torch.ops.aten.index_put.default(full, [None, None, arg3_1, arg5_1], add_1, True)
    add_3 = ip_result + ip2
    ip3 = torch.ops.aten.index_put.default(full, [None, None, arg6_1, arg4_1], mul_2, True)
    add_4 = add_3 + ip3
    ip4 = torch.ops.aten.index_put.default(full, [None, None, arg6_1, arg5_1], add_2, True)
    add_5 = add_4 + ip4
    conv_1 = add_5.to(torch.bfloat16)

    # Then BN "forward" pattern on arg7_1 with arg8/9/10/11
    sub = arg7_1 - arg8_1
    mul_3 = sub * arg9_1
    mul_4 = mul_3 * arg10_1.view(256, 1, 1)
    add_6 = mul_4 + arg11_1.view(256, 1, 1)
    conv_2 = add_6.to(torch.bfloat16)
    relu = torch.relu(conv_2)
    le = relu <= 0
    where_bf = torch.where(le, arg12_1, conv_1)

    # Prepare tensors for cuTile reductions/pointwise.
    C, H, W = 256, 80, 119
    spatial = H * W
    where_2d = where_bf.contiguous().view(C, spatial)
    arg7_2d = arg7_1.contiguous().view(C, spatial)

    BLOCK_C = 32
    BLOCK_S = 128  # power of 2, spatial=9520
    # Round up padded_S to a multiple of BLOCK_S AND make n_s_blocks power-of-2.
    def _next_pow2(n):
        p = 1
        while p < n:
            p *= 2
        return p
    n_blocks_raw = (spatial + BLOCK_S - 1) // BLOCK_S  # 75 for spatial=9520
    n_s_blocks = _next_pow2(n_blocks_raw)  # 128
    padded_S = n_s_blocks * BLOCK_S
    padded_where = torch.zeros((C, padded_S), device=device, dtype=torch.bfloat16)
    padded_where[:, :spatial].copy_(where_2d)
    padded_arg7 = torch.zeros((C, padded_S), device=device, dtype=torch.bfloat16)
    padded_arg7[:, :spatial].copy_(arg7_2d)

    squeeze = arg8_1.view(256)

    # Kernel 1: per-block partial sums for sum_1 and sum_2 (mirrors Triton's
    # _sum12_kernel per-channel reductions).
    stream = torch.cuda.current_stream()
    partial_s1 = torch.empty((C, n_s_blocks), device=device, dtype=torch.float32)
    partial_s2 = torch.empty((C, n_s_blocks), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), n_s_blocks, 1),
        _sum12_kernel,
        (padded_where, padded_arg7, squeeze, partial_s1, partial_s2,
         C, padded_S, BLOCK_C, BLOCK_S),
    )
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _channel_sum_kernel,
        (partial_s1, sum_1, n_s_blocks, BLOCK_C),
    )
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _channel_sum_kernel,
        (partial_s2, sum_2, n_s_blocks, BLOCK_C),
    )

    scale_recip = 0.0001050420168067227
    mul_6 = sum_1 * scale_recip
    mul_7 = sum_2 * scale_recip
    squeeze_1 = arg9_1.view(256)
    mul_8 = squeeze_1 * squeeze_1
    mul_9 = mul_7 * mul_8
    mul_10 = squeeze_1 * arg10_1  # scale
    mul_13 = sum_2 * squeeze_1

    # Kernel 2: fused BN-backward pointwise (mirrors Triton's _final_output_kernel).
    padded_out = torch.empty((C, padded_S), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), n_s_blocks, 1),
        _fused_kernel,
        (padded_where, padded_arg7, squeeze, mul_6, mul_9, mul_10,
         padded_out, C, padded_S, BLOCK_C, BLOCK_S),
    )
    conv5 = padded_out[:, :spatial].contiguous().view(1, C, H, W)

    # Kernel 3: per-block partial per-channel sum_3 (from the bf16 output).
    partial_s3 = torch.empty((C, n_s_blocks), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), n_s_blocks, 1),
        _sum3_kernel,
        (padded_out, partial_s3, spatial, BLOCK_C, BLOCK_S),
    )

    # Kernel 4: finalize sum_3 with bf16 rounding boundary.
    conv6 = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _channel_sum_bf16_kernel,
        (partial_s3, conv6, n_s_blocks, BLOCK_C),
    )

    return sum_1, mul_13, conv5, conv6
