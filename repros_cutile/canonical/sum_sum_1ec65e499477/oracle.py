"""cuTile port of sum_sum_1ec65e499477: ShuffleNet channel-shuffle + BN backward.

Matches Triton's three-kernel plan:
1. `_shuffle_materialize_kernel`: flat-index channel shuffle producer.
2. `_bn_reduce_kernel`: per-channel program reducing R = N*HW rows to sum_out
   and sum_centered.
3. `_bn_epilogue_kernel`: elementwise BN-backward tail writing dense_out and
   scale_grad (masked to the first 232 threads).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 232
OUT_C = 464
H = 7
W = 7
HW = H * W
R = N * HW
SHUFFLE_NUMEL = N * OUT_C * HW
BN_NUMEL = N * C * HW
SCALE = 0.00015943877551020407


@ct.kernel
def _shuffle_materialize_kernel(
    source_ptr,       # bf16 flat [SHUFFLE_NUMEL]  (physical channels-last)
    shuffled_out_ptr, # bf16 flat [SHUFFLE_NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    total = ct.full((BLOCK,), SHUFFLE_NUMEL, dtype=ct.int32)
    active = offsets < total

    w = offsets % 7
    h = (offsets // 7) % 7
    out_c = (offsets // 49) % 464
    n = offsets // 22736

    c232 = ct.full((BLOCK,), 232, dtype=ct.int32)
    src_c = ct.where(out_c < c232, out_c * 2, (out_c - c232) * 2 + 1)
    source_offsets = n * 22736 + src_c + h * 3248 + w * 464

    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    src_safe = ct.where(active, source_offsets, zero_i)
    values = ct.gather(source_ptr, src_safe, mask=active)
    ct.scatter(shuffled_out_ptr, offsets, values, mask=active)


@ct.kernel
def _bn_reduce_kernel(
    shuffle_source_ptr,   # bf16 flat [SHUFFLE_NUMEL]  (arg0 channels-last flat)
    bn_input_ptr,         # bf16 flat [BN_NUMEL]       (arg1 channels-last flat)
    mean_ptr,             # f32 [C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    bias_ptr,             # f32 [C]
    fill_ptr,             # bf16 [1]
    sum_out_ptr,          # f32 [C]
    sum_centered_ptr,     # f32 [C]
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    R_full = ct.full((BLOCK_R,), R, dtype=ct.int32)
    active = rows < R_full

    n = rows // HW
    spatial = rows - n * HW
    h = spatial // W
    w = spatial - h * W

    zero_i = ct.zeros((BLOCK_R,), dtype=ct.int32)
    bn_offsets = n * 11368 + c + h * 1624 + w * 232
    shuffle_offsets = n * 22736 + (c * 2 + 1) + h * 3248 + w * 464
    bn_off_safe = ct.where(active, bn_offsets, zero_i)
    sh_off_safe = ct.where(active, shuffle_offsets, zero_i)

    bn_input_bf = ct.gather(bn_input_ptr, bn_off_safe, mask=active)
    source_value_bf = ct.gather(shuffle_source_ptr, sh_off_safe, mask=active)
    bn_input = ct.astype(bn_input_bf, ct.float32)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_1 = ct.load(bias_ptr, index=(c,), shape=(1,))
    fill_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.reshape(mean_1, (1,))
    invstd = ct.reshape(invstd_1, (1,))
    weight = ct.reshape(weight_1, (1,))
    bias = ct.reshape(bias_1, (1,))
    fill_bf = ct.reshape(fill_1, (1,))

    centered = bn_input - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_check = ct.astype(affine_bf16, ct.float32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    fill_broadcast = ct.full((BLOCK_R,), 0.0, dtype=ct.bfloat16) + fill_bf
    where_bf16 = ct.where(affine_check <= zero_f, fill_broadcast, source_value_bf)
    where_f32 = ct.where(active, ct.astype(where_bf16, ct.float32), zero_f)
    centered_masked = ct.where(active, centered, zero_f)

    product = where_f32 * centered_masked
    sum_where = ct.sum(where_f32)
    sum_centered = ct.sum(product)

    sum_where_1 = ct.full((1,), sum_where, dtype=ct.float32)
    sum_centered_1 = ct.full((1,), sum_centered, dtype=ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=sum_where_1)
    ct.store(sum_centered_ptr, index=(c,), tile=sum_centered_1)


@ct.kernel
def _bn_epilogue_kernel(
    shuffle_source_ptr,   # bf16 flat [SHUFFLE_NUMEL]
    bn_input_ptr,         # bf16 flat [BN_NUMEL]
    mean_ptr,             # f32 [C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    bias_ptr,             # f32 [C]
    fill_ptr,             # bf16 [1]
    sum_out_ptr,          # f32 [C]
    sum_centered_ptr,     # f32 [C]
    scale_grad_ptr,       # f32 [C]
    dense_out_ptr,        # bf16 flat [BN_NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    linear = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    total = ct.full((BLOCK,), BN_NUMEL, dtype=ct.int32)
    active = linear < total

    c = linear % 232
    w = (linear // 232) % 7
    h = (linear // 1624) % 7
    n = linear // 11368

    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    shuffle_offsets = n * 22736 + (c * 2 + 1) + h * 3248 + w * 464
    lin_safe = ct.where(active, linear, zero_i)
    sh_off_safe = ct.where(active, shuffle_offsets, zero_i)
    c_safe = ct.where(active, c, zero_i)

    bn_input_bf = ct.gather(bn_input_ptr, lin_safe, mask=active)
    source_value_bf = ct.gather(shuffle_source_ptr, sh_off_safe, mask=active)
    bn_input = ct.astype(bn_input_bf, ct.float32)

    mean = ct.astype(ct.gather(mean_ptr, c_safe, mask=active), ct.float32)
    invstd = ct.astype(ct.gather(invstd_ptr, c_safe, mask=active), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_safe, mask=active), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_safe, mask=active), ct.float32)
    fill_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.reshape(fill_1, (1,))
    sum_where = ct.astype(ct.gather(sum_out_ptr, c_safe, mask=active), ct.float32)
    sum_centered = ct.astype(
        ct.gather(sum_centered_ptr, c_safe, mask=active), ct.float32
    )

    centered = bn_input - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_check = ct.astype(affine_bf16, ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    fill_broadcast = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16) + fill_bf
    where_bf16 = ct.where(affine_check <= zero_f, fill_broadcast, source_value_bf)
    where_f32 = ct.astype(where_bf16, ct.float32)

    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight

    after_variance = where_f32 - centered * variance_term
    after_mean = after_variance - mean_term
    dense_bf16 = ct.astype(after_mean * output_scale, ct.bfloat16)

    ct.scatter(dense_out_ptr, linear, dense_bf16, mask=active)

    # scale_grad[c]: first 232 threads only (linear < 232 means it doubles as c index).
    c_write_mask = active & (linear < ct.full((BLOCK,), 232, dtype=ct.int32))
    scale_grad_val = sum_centered * invstd
    ct.scatter(scale_grad_ptr, linear, scale_grad_val, mask=c_write_mask)


@oracle_impl(hardware="B200", point="0e23e8e0",
             SHUFFLE_BLOCK=1024, BLOCK_R=8192, EPILOGUE_BLOCK=256)
def oracle_forward(inputs, *, SHUFFLE_BLOCK: int, BLOCK_R: int, EPILOGUE_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, _, _ = inputs
    device = arg0_1.device

    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    sum_centered = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )

    # 1D flat views of physical storage (channels-last -> memory order).
    arg0_flat = torch.as_strided(arg0_1, (SHUFFLE_NUMEL,), (1,))
    arg1_flat = torch.as_strided(arg1_1, (BN_NUMEL,), (1,))
    shuffled_flat = torch.as_strided(shuffled, (SHUFFLE_NUMEL,), (1,))
    dense_flat = torch.as_strided(dense_out, (BN_NUMEL,), (1,))

    mean_1d = arg2_1.view(C)
    invstd_1d = arg3_1.view(C)
    weight_1d = arg4_1
    bias_1d = arg5_1
    fill_1d = arg6_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(SHUFFLE_NUMEL, SHUFFLE_BLOCK), 1, 1),
        _shuffle_materialize_kernel,
        (arg0_flat, shuffled_flat, SHUFFLE_BLOCK),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_reduce_kernel,
        (arg0_flat, arg1_flat, mean_1d, invstd_1d, weight_1d, bias_1d,
         fill_1d, sum_out, sum_centered, BLOCK_R),
    )
    ct.launch(
        stream,
        (ct.cdiv(BN_NUMEL, EPILOGUE_BLOCK), 1, 1),
        _bn_epilogue_kernel,
        (arg0_flat, arg1_flat, mean_1d, invstd_1d, weight_1d, bias_1d,
         fill_1d, sum_out, sum_centered, scale_grad, dense_flat, EPILOGUE_BLOCK),
    )

    return shuffled, sum_out, scale_grad, dense_out
