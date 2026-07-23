"""cuTile port of sum_sum_052b3bc6efa8 (SCATTER_REDUCE): Inception max-pool
backward scatter + BN backward with 147x147 spatial dims.

Uses torch.scatter_add for the max-pool-backward reverse-scatter (cuTile lacks
efficient atomic-add). All BN-backward channel reductions and the epilogue
happen in cuTile kernels mirroring Triton's `_partial_reduce_kernel`,
`_finalize_kernel`, and `_epilogue_kernel` structure.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 64
POOL_H = 73
POOL_W = 73
POOL_HW = 5329
H = 147
W = 147
HW = 21609
SPATIAL = N * HW  # 2765952
SCALE = 3.6153917349252627e-07

BLOCK_K = 256
NUM_TILES = (SPATIAL + BLOCK_K - 1) // BLOCK_K  # 10805


def _next_p2(v):
    r = 1
    while r < int(v):
        r <<= 1
    return r


BLOCK_TILES = _next_p2(NUM_TILES)  # 16384


@ct.kernel
def _partial_reduce_kernel(
    source_ptr,      # bf16 flat channels-last physical [N, HW, C]
    x_ptr,           # bf16 flat channels-last physical [N, HW, C]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    fill_ptr,        # bf16 [1]
    partial_sum0_ptr,  # f32 [C, NUM_TILES]
    partial_sum1_ptr,  # f32 [C, NUM_TILES]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    SPATIAL_: ct.Constant[int],
    NUM_TILES_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    c_ids = c_block * BLOCK_C_ + ct.arange(BLOCK_C_, dtype=ct.int32)
    k_ids = k_block * BLOCK_K_ + ct.arange(BLOCK_K_, dtype=ct.int32)
    c_valid = c_ids < C_
    k_valid = k_ids < SPATIAL_

    c_2d = ct.reshape(c_ids, (1, BLOCK_C_))
    k_2d = ct.reshape(k_ids, (BLOCK_K_, 1))
    c_valid_2d = ct.reshape(c_valid, (1, BLOCK_C_))
    k_valid_2d = ct.reshape(k_valid, (BLOCK_K_, 1))
    active = k_valid_2d & c_valid_2d

    # Compute flat offset: k * C + c (since layout is [SPATIAL, C] contiguous)
    offsets = k_2d * C_ + c_2d

    scatter = ct.gather(source_ptr, offsets, mask=active)
    x = ct.gather(x_ptr, offsets, mask=active)
    mean = ct.astype(ct.gather(mean_ptr, c_ids), ct.float32)
    invstd = ct.astype(ct.gather(invstd_ptr, c_ids), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_ids), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_ids), ct.float32)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    mean_2d = ct.reshape(mean, (1, BLOCK_C_))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C_))
    weight_2d = ct.reshape(weight, (1, BLOCK_C_))
    bias_2d = ct.reshape(bias, (1, BLOCK_C_))
    fill_f = ct.astype(fill, ct.float32)
    fill_bc = ct.broadcast_to(ct.reshape(fill_f, (1, 1)), (BLOCK_K_, BLOCK_C_))

    x_f = ct.astype(x, ct.float32)
    scatter_f_raw = ct.astype(scatter, ct.float32)
    # Simulate the Triton bf16 roundtrip on scatter.
    scatter_f = ct.astype(ct.astype(scatter_f_raw, ct.bfloat16), ct.float32)
    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    scaled = normalized * weight_2d
    affine = scaled + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_bf_f = ct.astype(affine_bf, ct.float32)
    where_value = ct.where(affine_bf_f <= 0.0, fill_bc, scatter_f)
    where_value = ct.where(active, where_value, 0.0)
    centered_masked = ct.where(active, centered, 0.0)

    sum0 = ct.sum(where_value, axis=0)  # (BLOCK_C,)
    sum1 = ct.sum(where_value * centered_masked, axis=0)

    # Partial storage layout is [C, NUM_TILES]: partial[c, k_block]
    partial_offsets = c_ids * NUM_TILES_ + k_block
    ct.scatter(partial_sum0_ptr, partial_offsets, sum0, mask=c_valid)
    ct.scatter(partial_sum1_ptr, partial_offsets, sum1, mask=c_valid)


@ct.kernel
def _finalize_kernel(
    partial_sum0_ptr,  # f32 [C, NUM_TILES]
    partial_sum1_ptr,  # f32 [C, NUM_TILES]
    invstd_ptr,        # f32 [C]
    sum0_ptr,          # f32 [C]
    sum1_ptr,          # f32 [C]
    out0_ptr,          # f32 [C]
    out1_ptr,          # f32 [C]
    NUM_TILES_: ct.Constant[int],
    BLOCK_TILES_: ct.Constant[int],
):
    c = ct.bid(0)
    parts0 = ct.load(partial_sum0_ptr, index=(c, 0),
                     shape=(1, BLOCK_TILES_),
                     padding_mode=ct.PaddingMode.ZERO)
    parts1 = ct.load(partial_sum1_ptr, index=(c, 0),
                     shape=(1, BLOCK_TILES_),
                     padding_mode=ct.PaddingMode.ZERO)
    idx = ct.arange(BLOCK_TILES_, dtype=ct.int32)
    valid = ct.reshape(idx < NUM_TILES_, (1, BLOCK_TILES_))
    parts0_masked = ct.where(valid, parts0, 0.0)
    parts1_masked = ct.where(valid, parts1, 0.0)
    total0 = ct.sum(parts0_masked)
    total1 = ct.sum(parts1_masked)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c,), shape=(1,)), ct.float32)
    invstd_scalar = ct.reshape(invstd, (1,))
    ct.store(sum0_ptr, index=(c,), tile=ct.reshape(total0, (1,)))
    ct.store(sum1_ptr, index=(c,), tile=ct.reshape(total1, (1,)))
    ct.store(out0_ptr, index=(c,), tile=ct.reshape(total0, (1,)))
    ct.store(out1_ptr, index=(c,), tile=ct.reshape(total1, (1,)) * invstd_scalar)


@ct.kernel
def _epilogue_kernel(
    source_ptr,     # bf16 flat channels-last [SPATIAL, C]
    x_ptr,          # bf16 flat channels-last [SPATIAL, C]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    fill_ptr,       # bf16 [1]
    sum0_ptr,       # f32 [C]
    sum1_ptr,       # f32 [C]
    out_ptr,        # bf16 flat channels-last [SPATIAL, C]
    C_: ct.Constant[int],
    SPATIAL_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    c_ids = c_block * BLOCK_C_ + ct.arange(BLOCK_C_, dtype=ct.int32)
    k_ids = k_block * BLOCK_K_ + ct.arange(BLOCK_K_, dtype=ct.int32)
    c_valid = c_ids < C_
    k_valid = k_ids < SPATIAL_

    c_2d = ct.reshape(c_ids, (1, BLOCK_C_))
    k_2d = ct.reshape(k_ids, (BLOCK_K_, 1))
    c_valid_2d = ct.reshape(c_valid, (1, BLOCK_C_))
    k_valid_2d = ct.reshape(k_valid, (BLOCK_K_, 1))
    active = k_valid_2d & c_valid_2d

    offsets = k_2d * C_ + c_2d

    scatter = ct.gather(source_ptr, offsets, mask=active)
    x = ct.gather(x_ptr, offsets, mask=active)
    mean = ct.astype(ct.gather(mean_ptr, c_ids), ct.float32)
    invstd = ct.astype(ct.gather(invstd_ptr, c_ids), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_ids), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, c_ids), ct.float32)
    total0 = ct.astype(ct.gather(sum0_ptr, c_ids), ct.float32)
    total1 = ct.astype(ct.gather(sum1_ptr, c_ids), ct.float32)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    mean_2d = ct.reshape(mean, (1, BLOCK_C_))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C_))
    weight_2d = ct.reshape(weight, (1, BLOCK_C_))
    bias_2d = ct.reshape(bias, (1, BLOCK_C_))
    total0_2d = ct.reshape(total0, (1, BLOCK_C_))
    total1_2d = ct.reshape(total1, (1, BLOCK_C_))
    fill_f = ct.astype(fill, ct.float32)
    fill_bc = ct.broadcast_to(ct.reshape(fill_f, (1, 1)), (BLOCK_K_, BLOCK_C_))

    x_f = ct.astype(x, ct.float32)
    scatter_f_raw = ct.astype(scatter, ct.float32)
    scatter_f = ct.astype(ct.astype(scatter_f_raw, ct.bfloat16), ct.float32)
    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    scaled = normalized * weight_2d
    affine = scaled + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_bf_f = ct.astype(affine_bf, ct.float32)
    where_value = ct.where(affine_bf_f <= 0.0, fill_bc, scatter_f)

    mean_term = total0_2d * SCALE_
    var_scaled = (total1_2d * SCALE_) * (invstd_2d * invstd_2d)
    without_var = where_value - centered * var_scaled
    without_mean = without_var - mean_term
    output_scale = invstd_2d * weight_2d
    out = without_mean * output_scale
    ct.scatter(out_ptr, offsets, ct.astype(out, ct.bfloat16), mask=active)


@oracle_impl(
    hardware="B200",
    point="f00fbabc",
    BLOCK_C=32,
    BLOCK_K=256,
    ZERO_BLOCK=1024,
    SCATTER_BLOCK=256,
)
def oracle_forward(inputs, **_kwargs):
    pooled, offsets_i8, x, mean, invstd, weight, bias, scalar = inputs[:8]
    device = x.device

    # 1) Use torch.scatter_add for reverse-scatter (cuTile lacks atomic-add).
    kernel_size = [3, 3]
    input_size = [H, W]
    stride = [2, 2]
    padding = [0, 0]
    dilation = [1, 1]
    indices_i64 = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        offsets_i8, kernel_size, input_size, stride, padding, dilation,
    )
    indices_2d = indices_i64.contiguous().view(N * C, POOL_HW)

    pooled_contig = pooled.contiguous()
    pooled_2d_f32 = pooled_contig.view(N * C, POOL_HW).to(torch.float32)
    scatter_f32 = torch.zeros((N * C, HW), device=device, dtype=torch.float32)
    scatter_f32.scatter_add_(1, indices_2d, pooled_2d_f32)

    # source needs to be in channels-last physical layout matching x.
    scatter_bf16_nchw = scatter_f32.view(N, C, H, W).to(torch.bfloat16)
    scatter_bf16_cl = scatter_bf16_nchw.contiguous(memory_format=torch.channels_last)
    # Flat channels-last physical view: (SPATIAL, C) i.e., NHWC.
    source_flat = scatter_bf16_cl.as_strided((SPATIAL, C), (C, 1))

    # x is channels-last strided: (C*HW, 1, W*C, C). Flat NHWC view:
    x_flat = x.as_strided((SPATIAL, C), (C, 1))

    partial_sum0 = torch.empty((C, NUM_TILES), device=device, dtype=torch.float32)
    partial_sum1 = torch.empty((C, NUM_TILES), device=device, dtype=torch.float32)
    sum0 = torch.empty((C,), device=device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=device, dtype=torch.float32)
    out0 = torch.empty((C,), device=device, dtype=torch.float32)
    out1 = torch.empty((C,), device=device, dtype=torch.float32)
    out2 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    out_flat = out2.as_strided((SPATIAL, C), (C, 1))

    scalar_flat = scalar.view(1)

    BLOCK_C = 32
    stream = torch.cuda.current_stream()
    mean_1d = mean.reshape(C)
    invstd_1d = invstd.reshape(C)
    weight_1d = weight.reshape(C)
    bias_1d = bias.reshape(C)
    # Kernel 1: partial reduce
    ct.launch(
        stream, (C // BLOCK_C, NUM_TILES, 1),
        _partial_reduce_kernel,
        (source_flat.reshape(-1), x_flat.reshape(-1),
         mean_1d, invstd_1d, weight_1d, bias_1d, scalar_flat,
         partial_sum0.view(-1), partial_sum1.view(-1),
         C, HW, SPATIAL, NUM_TILES, BLOCK_C, BLOCK_K),
    )
    # Kernel 2: finalize
    ct.launch(
        stream, (C, 1, 1),
        _finalize_kernel,
        (partial_sum0, partial_sum1, invstd_1d,
         sum0, sum1, out0, out1,
         NUM_TILES, BLOCK_TILES),
    )
    # Kernel 3: epilogue
    ct.launch(
        stream, (C // BLOCK_C, NUM_TILES, 1),
        _epilogue_kernel,
        (source_flat.reshape(-1), x_flat.reshape(-1),
         mean_1d, invstd_1d, weight_1d, bias_1d, scalar_flat,
         sum0, sum1, out_flat.reshape(-1),
         C, SPATIAL, BLOCK_C, BLOCK_K, SCALE),
    )

    return out0, out1, out2
