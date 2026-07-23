"""cuTile port of sum_sum_592ede6a40e5: Visformer BN-backward tail.

Mirrors Triton's three-kernel COOPERATIVE_SPLIT_K structure:
  * `_partial_reduce_kernel` — per (c_block, k_block) tile does the masked
    producer + centered-dot in-kernel and stores per-channel partials.
  * `_finalize_kernel` — per channel reduces partial rows into the returned
    sum, dot, and invstd-scaled dot.
  * `_epilogue_kernel` — flat channels-last epilogue that recomputes the
    masked producer and writes the returned bf16 output.

Inputs are channels-last [N,C,H,W] with C as the stride-1 axis, so the flat
address is `k * C + c` where `k = n * HW + spatial`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 32
H = 112
W = 112
HW = H * W
K_TOTAL = N * HW
TOTAL = K_TOTAL * C
SCALE = 6.228077168367346e-07


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,             # bf16 [K_TOTAL, C]
    source_ptr,           # bf16 [K_TOTAL, C]
    centered_source_ptr,  # bf16 [K_TOTAL, C]
    mean_ptr,             # f32  [C]
    partial_sum_ptr,      # f32  [C, NUM_K_BLOCKS]
    partial_dot_ptr,      # f32  [C, NUM_K_BLOCKS]
    NUM_K_BLOCKS: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    k = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    c = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    k_2d = ct.reshape(k, (BLOCK_K, 1))
    c_2d = ct.reshape(c, (1, BLOCK_C))
    active = (k_2d < K_TOTAL_) & (c_2d < C_)

    mask_bf16 = ct.load(
        mask_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    source_bf16 = ct.load(
        source_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered_source_bf16 = ct.load(
        centered_source_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean_1d = ct.load(
        mean_ptr, index=(c_block,), shape=(BLOCK_C,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    mask_f32 = ct.astype(mask_bf16, ct.float32)
    source_f32 = ct.astype(source_bf16, ct.float32)
    centered_source_f32 = ct.astype(centered_source_bf16, ct.float32)

    zero_tile = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32)
    producer = ct.where(mask_f32 <= 0.0, zero_tile, source_f32)
    producer = ct.where(active, producer, zero_tile)

    mean_2d = ct.reshape(mean_1d, (1, BLOCK_C))
    centered = centered_source_f32 - mean_2d
    dot = ct.where(active, producer * centered, zero_tile)

    partial_sum = ct.sum(producer, axis=0)   # (BLOCK_C,)
    partial_dot = ct.sum(dot, axis=0)        # (BLOCK_C,)

    ct.store(
        partial_sum_ptr,
        index=(c_block, k_block),
        tile=ct.reshape(partial_sum, (BLOCK_C, 1)),
    )
    ct.store(
        partial_dot_ptr,
        index=(c_block, k_block),
        tile=ct.reshape(partial_dot, (BLOCK_C, 1)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [C, NUM_K_BLOCKS]
    partial_dot_ptr,   # f32 [C, NUM_K_BLOCKS]
    invstd_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    dot_tmp_ptr,       # f32 [C]
    scaled_dot_ptr,    # f32 [C]
    NUM_K_BLOCKS: ct.Constant[int],
    BLOCK_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)

    partial_sum_tile = ct.load(
        partial_sum_ptr, index=(c, 0), shape=(1, BLOCK_BLOCKS),
        padding_mode=ct.PaddingMode.ZERO,
    )
    partial_dot_tile = ct.load(
        partial_dot_ptr, index=(c, 0), shape=(1, BLOCK_BLOCKS),
        padding_mode=ct.PaddingMode.ZERO,
    )

    block_idx = ct.arange(BLOCK_BLOCKS, dtype=ct.int32)
    active = ct.reshape(block_idx < NUM_K_BLOCKS, (1, BLOCK_BLOCKS))
    zero_row = ct.zeros((1, BLOCK_BLOCKS), dtype=ct.float32)
    partial_sum_masked = ct.where(active, partial_sum_tile, zero_row)
    partial_dot_masked = ct.where(active, partial_dot_tile, zero_row)

    sum_value = ct.sum(partial_sum_masked)
    dot_value = ct.sum(partial_dot_masked)

    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    invstd_scalar = ct.reshape(invstd, ())

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(dot_tmp_ptr, index=(c,), tile=ct.reshape(dot_value, (1,)))
    ct.store(
        scaled_dot_ptr,
        index=(c,),
        tile=ct.reshape(dot_value * invstd_scalar, (1,)),
    )


@ct.kernel
def _epilogue_kernel(
    mask_ptr,             # bf16 [TOTAL] flat channels-last
    source_ptr,           # bf16 [TOTAL]
    centered_source_ptr,  # bf16 [TOTAL]
    mean_ptr,             # f32 [C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    sum_ptr,              # f32 [C]
    dot_ptr,              # f32 [C]
    out_ptr,              # bf16 [TOTAL]
    TOTAL_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_idx = offsets % C_

    mask_bf16 = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    source_bf16 = ct.load(source_ptr, index=(pid,), shape=(BLOCK,))
    centered_source_bf16 = ct.load(
        centered_source_ptr, index=(pid,), shape=(BLOCK,)
    )

    mean = ct.astype(ct.gather(mean_ptr, c_idx), ct.float32)
    invstd = ct.astype(ct.gather(invstd_ptr, c_idx), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, c_idx), ct.float32)
    sum_value = ct.astype(ct.gather(sum_ptr, c_idx), ct.float32)
    dot_value = ct.astype(ct.gather(dot_ptr, c_idx), ct.float32)

    mask_f32 = ct.astype(mask_bf16, ct.float32)
    source_f32 = ct.astype(source_bf16, ct.float32)
    centered_source_f32 = ct.astype(centered_source_bf16, ct.float32)
    zero_tile = ct.zeros((BLOCK,), dtype=ct.float32)
    producer = ct.where(mask_f32 <= 0.0, zero_tile, source_f32)
    centered = centered_source_f32 - mean

    mean_term = sum_value * SCALE
    dot_scaled = dot_value * SCALE
    variance_term = dot_scaled * (invstd * invstd)
    output_scale = invstd * weight
    after_variance = producer - centered * variance_term
    after_mean = after_variance - mean_term
    out = ct.astype(after_mean * output_scale, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="bf7e56a6", BLOCK_K=1024, BLOCK_C=8, EPILOGUE_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, EPILOGUE_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg1_1.device
    num_k_blocks = (K_TOTAL + BLOCK_K - 1) // BLOCK_K

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    partial_sum = torch.empty_strided(
        (C, num_k_blocks), (num_k_blocks, 1),
        device=device, dtype=torch.float32,
    )
    partial_dot = torch.empty_strided(
        (C, num_k_blocks), (num_k_blocks, 1),
        device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=device,
        dtype=torch.bfloat16,
    )

    # Flat channels-last views: physical storage index is `k * C + c` since C
    # is the stride-1 axis. `torch.as_strided(..., (K_TOTAL, C), (C, 1))`
    # aliases the same storage in that logical layout.
    mask_2d = torch.as_strided(arg0_1, (K_TOTAL, C), (C, 1))
    source_2d = torch.as_strided(arg1_1, (K_TOTAL, C), (C, 1))
    centered_source_2d = torch.as_strided(arg2_1, (K_TOTAL, C), (C, 1))
    mask_flat = torch.as_strided(arg0_1, (TOTAL,), (1,))
    source_flat = torch.as_strided(arg1_1, (TOTAL,), (1,))
    centered_source_flat = torch.as_strided(arg2_1, (TOTAL,), (1,))
    out_flat = torch.as_strided(out, (TOTAL,), (1,))

    mean_1d = arg3_1.view(C)
    invstd_1d = arg4_1.view(C)
    weight_1d = arg5_1.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), num_k_blocks, 1),
        _partial_reduce_kernel,
        (mask_2d, source_2d, centered_source_2d, mean_1d,
         partial_sum, partial_dot,
         num_k_blocks, K_TOTAL, C, BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d,
         sum_out, dot_tmp, scaled_dot,
         num_k_blocks, _next_power_of_2(num_k_blocks)),
    )
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, EPILOGUE_BLOCK), 1, 1),
        _epilogue_kernel,
        (mask_flat, source_flat, centered_source_flat,
         mean_1d, invstd_1d, weight_1d, sum_out, dot_tmp,
         out_flat,
         TOTAL, C, EPILOGUE_BLOCK),
    )
    return sum_out, scaled_dot, out
