"""cuTile port of sum_sum_ab9fc4091e40: DenseNet BN-backward tail.

Matches Triton's 3-kernel structure exactly: partial reduce (split-K over
N*HW), per-channel finalize, and the epilogue (BN-backward gradient +
5-residual slice add). All math ops (where mask select, sub, mul, sum
reductions, scale math) live inside cuTile kernels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 96
H = 56
W = 56
HW = H * W        # 3136
K_TOTAL = N * HW  # 12544
SCALE = 7.971938775510203e-05
SLICE_START = 64
SLICE_C = 32
RESIDUAL_CHANNELS = (256, 224, 192, 160, 128)


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,             # bf16 [N, C, HW]
    fill_ptr,             # bf16 [1]
    source_ptr,           # bf16 [N, C, HW]
    centered_source_ptr,  # bf16 [N, C, HW]
    mean_ptr,             # f32  [C]
    partial_sum_ptr,      # f32  [NUM_BLOCKS, C]
    partial_dot_ptr,      # f32  [NUM_BLOCKS, C]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    K_TOTAL_C: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    """Grid: (C, num_blocks, 1)."""
    c = ct.bid(0)
    block = ct.bid(1)
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    k_global = block * BLOCK_K + k_idx
    active = k_global < K_TOTAL_C
    n = k_global // HW_C
    spatial = k_global - n * HW_C
    c_bc = ct.full((BLOCK_K,), c, dtype=ct.int32)

    mask_v = ct.gather(mask_ptr, (n, c_bc, spatial), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n, c_bc, spatial), mask=active,
                         padding_value=ct.bfloat16(0.0))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_K,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    grad_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    grad = ct.astype(grad_bf, ct.float32)

    centered_src = ct.astype(
        ct.gather(centered_source_ptr, (n, c_bc, spatial), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    centered = centered_src - ct.broadcast_to(mean, (BLOCK_K,))

    grad = ct.where(active, grad, 0.0)
    centered = ct.where(active, centered, 0.0)

    ps = ct.sum(grad)
    pd = ct.sum(grad * centered)
    ct.store(partial_sum_ptr, index=(block, c), tile=ct.reshape(ps, (1, 1)))
    ct.store(partial_dot_ptr, index=(block, c), tile=ct.reshape(pd, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [NUM_BLOCKS, C]
    partial_dot_ptr,   # f32 [NUM_BLOCKS, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scale_grad_ptr,    # f32 [C]
    mean_term_ptr,     # f32 [C]
    variance_ptr,      # f32 [C]
    output_scale_ptr,  # f32 [C]
    C_C: ct.Constant[int],
    NUM_BLOCKS: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_BLOCKS_: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    c_block = ct.bid(0)
    part_sum = ct.astype(
        ct.load(partial_sum_ptr, index=(0, c_block),
                shape=(BLOCK_BLOCKS_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    part_dot = ct.astype(
        ct.load(partial_dot_ptr, index=(0, c_block),
                shape=(BLOCK_BLOCKS_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    sum_v = ct.sum(part_sum, axis=0)
    dot_v = ct.sum(part_dot, axis=0)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    scale_grad = dot_v * invstd
    mean_term = sum_v * SCALE_C
    variance = (dot_v * SCALE_C) * (invstd * invstd)
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_v)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(variance_ptr, index=(c_block,), tile=variance)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    residual0_ptr,        # bf16 [N, 256, HW]
    residual1_ptr,        # bf16 [N, 224, HW]
    residual2_ptr,        # bf16 [N, 192, HW]
    residual3_ptr,        # bf16 [N, 160, HW]
    residual4_ptr,        # bf16 [N, 128, HW]
    mask_ptr,             # bf16 [N, C, HW]
    fill_ptr,             # bf16 [1]
    source_ptr,           # bf16 [N, C, HW]
    centered_source_ptr,  # bf16 [N, C, HW]
    mean_ptr,             # f32  [C]
    mean_term_ptr,        # f32  [C]
    variance_ptr,         # f32  [C]
    output_scale_ptr,     # f32  [C]
    dense_out_ptr,        # bf16 [N, C, HW]
    add_out_ptr,          # bf16 [N, SLICE_C, HW]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    K_TOTAL_C: ct.Constant[int],
    SLICE_START_C: ct.Constant[int],
    SLICE_C_C: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    """Grid: (C, num_blocks, 1). Writes dense_out for all c and add_out for
    c >= SLICE_START (into channel slice_c = c - SLICE_START)."""
    c = ct.bid(0)
    block = ct.bid(1)
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    k_global = block * BLOCK_K + k_idx
    active = k_global < K_TOTAL_C
    n = k_global // HW_C
    spatial = k_global - n * HW_C
    c_bc = ct.full((BLOCK_K,), c, dtype=ct.int32)

    mask_v = ct.gather(mask_ptr, (n, c_bc, spatial), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n, c_bc, spatial), mask=active,
                         padding_value=ct.bfloat16(0.0))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_K,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    grad_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    grad = ct.astype(grad_bf, ct.float32)

    centered_src = ct.astype(
        ct.gather(centered_source_ptr, (n, c_bc, spatial), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_term = ct.astype(ct.load(mean_term_ptr, index=(c,), shape=(1,)),
                          ct.float32)
    variance = ct.astype(ct.load(variance_ptr, index=(c,), shape=(1,)),
                         ct.float32)
    output_scale = ct.astype(ct.load(output_scale_ptr, index=(c,), shape=(1,)),
                             ct.float32)
    centered = centered_src - ct.broadcast_to(mean, (BLOCK_K,))
    variance_bc = ct.broadcast_to(variance, (BLOCK_K,))
    mean_term_bc = ct.broadcast_to(mean_term, (BLOCK_K,))
    output_scale_bc = ct.broadcast_to(output_scale, (BLOCK_K,))

    corrected = grad - centered * variance_bc
    centered_grad = corrected - mean_term_bc
    dense_bf16 = ct.astype(centered_grad * output_scale_bc, ct.bfloat16)
    ct.scatter(dense_out_ptr, (n, c_bc, spatial), dense_bf16, mask=active)

    # Residual add for c >= SLICE_START.  All residuals are dense NCHW; we
    # slice channels [SLICE_START:SLICE_START+SLICE_C] from each.  Match
    # Triton's Repro-visible sequential bf16 add chain by converting each
    # via f32 add (RTNE-equivalent) and staying in bf16 after each step.
    in_slice = c >= SLICE_START_C
    slice_c_scalar = c - SLICE_START_C
    slice_c_bc = ct.full((BLOCK_K,), slice_c_scalar, dtype=ct.int32)
    add_active = active & in_slice
    # Load residuals — gather at (n, c, spatial) using c=full channel index c.
    v0 = ct.gather(residual0_ptr, (n, c_bc, spatial), mask=add_active,
                   padding_value=ct.bfloat16(0.0))
    v1 = ct.gather(residual1_ptr, (n, c_bc, spatial), mask=add_active,
                   padding_value=ct.bfloat16(0.0))
    v2 = ct.gather(residual2_ptr, (n, c_bc, spatial), mask=add_active,
                   padding_value=ct.bfloat16(0.0))
    v3 = ct.gather(residual3_ptr, (n, c_bc, spatial), mask=add_active,
                   padding_value=ct.bfloat16(0.0))
    v4 = ct.gather(residual4_ptr, (n, c_bc, spatial), mask=add_active,
                   padding_value=ct.bfloat16(0.0))

    def _bf16_add(a, b):
        return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
                          ct.bfloat16)

    residual = _bf16_add(v0, v1)
    residual = _bf16_add(residual, v2)
    residual = _bf16_add(residual, v3)
    residual = _bf16_add(residual, v4)
    add_value = _bf16_add(residual, dense_bf16)

    ct.scatter(add_out_ptr, (n, slice_c_bc, spatial), add_value, mask=add_active)


@oracle_impl(hardware="B200", point="72d2f85d", BLOCK_K=8192)
def oracle_forward(inputs, *, BLOCK_K):
    (
        arg0_1,  # bf16 (N, 256, H, W)   residual0
        arg1_1,  # bf16 (N, 224, H, W)   residual1
        arg2_1,  # bf16 (N, 192, H, W)   residual2
        arg3_1,  # bf16 (N, 160, H, W)   residual3
        arg4_1,  # bf16 (N, 128, H, W)   residual4
        arg5_1,  # bf16 (N, 96, H, W)    mask_input
        arg6_1,  # bf16 scalar           fill
        arg7_1,  # bf16 (N, 96, H, W)    source
        arg8_1,  # bf16 (N, 96, H, W)    centered_source
        arg9_1,  # f32  (1, 96, 1, 1)    mean
        arg10_1, # f32  (96,)            invstd
        arg11_1, # f32  (96,)            weight
    ) = inputs
    device = arg7_1.device

    num_blocks = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    block_blocks_pow2 = 1 << (num_blocks - 1).bit_length()

    partial_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_blocks, C), device=device, dtype=torch.float32)

    mask_3d = arg5_1.view(N, C, HW)
    source_3d = arg7_1.view(N, C, HW)
    centered_3d = arg8_1.view(N, C, HW)
    fill_1d = arg6_1.view(1)
    mean_1d = arg9_1.view(C)
    invstd_1d = arg10_1.view(C)
    weight_1d = arg11_1.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, num_blocks, 1), _partial_reduce_kernel,
        (mask_3d, fill_1d, source_3d, centered_3d, mean_1d,
         partial_sum, partial_dot,
         C, HW, K_TOTAL, BLOCK_K),
    )

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    variance = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    BLOCK_C = 8
    assert C % BLOCK_C == 0
    ct.launch(
        stream, (C // BLOCK_C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scale_grad, mean_term, variance, output_scale,
         C, num_blocks, BLOCK_C, block_blocks_pow2, SCALE),
    )

    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W), (SLICE_C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    residual0_3d = arg0_1.view(N, RESIDUAL_CHANNELS[0], HW)
    residual1_3d = arg1_1.view(N, RESIDUAL_CHANNELS[1], HW)
    residual2_3d = arg2_1.view(N, RESIDUAL_CHANNELS[2], HW)
    residual3_3d = arg3_1.view(N, RESIDUAL_CHANNELS[3], HW)
    residual4_3d = arg4_1.view(N, RESIDUAL_CHANNELS[4], HW)
    dense_out_3d = dense_out.view(N, C, HW)
    add_out_3d = add_out.view(N, SLICE_C, HW)

    ct.launch(
        stream, (C, num_blocks, 1), _epilogue_kernel,
        (residual0_3d, residual1_3d, residual2_3d, residual3_3d, residual4_3d,
         mask_3d, fill_1d, source_3d, centered_3d, mean_1d,
         mean_term, variance, output_scale, dense_out_3d, add_out_3d,
         C, HW, K_TOTAL, SLICE_START, SLICE_C, BLOCK_K),
    )
    return sum_out, scale_grad, dense_out, add_out
