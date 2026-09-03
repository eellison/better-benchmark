"""cuTile port of sum_sum_c0554ca7f4d6: MobileViT SiLU-backward + BN-backward tail.

Mirrors Triton's 3-kernel structure: partial_reduce (produces producer + partial
per-channel sums), finalize (reduces partials into per-channel scalars), and
epilogue (writes the dense bf16 output using the finalized scalars).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 160
H = 8
W = 8
HW = H * W
K_TOTAL = N * HW  # 8192
TOTAL = K_TOTAL * C  # 1310720
REDUCE_SCALE = 0.0001220703125


def _next_power_of_2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    upstream_ptr,     # bf16 (K_TOTAL, C) contiguous
    x_ptr,            # bf16 (K_TOTAL, C) contiguous
    mean_ptr,         # f32 (C,)
    invstd_ptr,       # f32 (C,)
    weight_ptr,       # f32 (C,)
    bias_ptr,         # f32 (C,)
    producer_ptr,     # bf16 (K_TOTAL, C) contiguous
    partial_sum_ptr,  # f32 (num_k_tiles, C)
    partial_dot_ptr,  # f32 (num_k_tiles, C)
    GROUP_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    # Loads
    upstream = ct.load(upstream_ptr, index=(k_block, c_block), shape=(GROUP_K, BLOCK_C))
    x = ct.load(x_ptr, index=(k_block, c_block), shape=(GROUP_K, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))

    # Broadcast per-channel to (1, BLOCK_C)
    mean_row = ct.reshape(mean, (1, BLOCK_C))
    invstd_row = ct.reshape(invstd, (1, BLOCK_C))
    weight_row = ct.reshape(weight, (1, BLOCK_C))
    bias_row = ct.reshape(bias, (1, BLOCK_C))

    x_f = ct.astype(x, ct.float32)
    upstream_f = ct.astype(upstream, ct.float32)
    centered = x_f - mean_row
    normalized = centered * invstd_row
    affine = normalized * weight_row + bias_row
    # Round-trip bf16 to match Triton's affine.to(bf16, rtne).to(f32)
    rounded_affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    exp_neg = ct.exp(-rounded_affine)
    sigmoid = 1.0 / (exp_neg + 1.0)
    grad_sigmoid = upstream_f * sigmoid
    one_minus = 1.0 - sigmoid
    affine_tail = rounded_affine * one_minus + 1.0
    producer_f = grad_sigmoid * affine_tail
    producer_bf = ct.astype(producer_f, ct.bfloat16)
    ct.store(producer_ptr, index=(k_block, c_block), tile=producer_bf)

    # Reduce along k axis with producer cast back to f32 (matches Triton).
    producer_f32 = ct.astype(producer_bf, ct.float32)
    dot_val = producer_f32 * centered
    partial_sum = ct.sum(producer_f32, axis=0)  # (BLOCK_C,)
    partial_dot = ct.sum(dot_val, axis=0)

    ct.store(partial_sum_ptr, index=(k_block, c_block),
             tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_dot_ptr, index=(k_block, c_block),
             tile=ct.reshape(partial_dot, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (NUM_K_TILES, C)
    partial_dot_ptr,   # f32 (NUM_K_TILES, C)
    invstd_ptr,        # f32 (C,)
    sum_out_ptr,       # f32 (C,)
    dot_tmp_ptr,       # f32 (C,)
    scaled_dot_ptr,    # f32 (C,)
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    partials_sum = ct.load(
        partial_sum_ptr, index=(0, c), shape=(BLOCK_TILES, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    partials_dot = ct.load(
        partial_dot_ptr, index=(0, c), shape=(BLOCK_TILES, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sum_val = ct.sum(partials_sum, axis=0)  # (1,)
    dot_val = ct.sum(partials_dot, axis=0)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    scaled = dot_val * invstd

    ct.store(sum_out_ptr, index=(c,), tile=sum_val)
    ct.store(dot_tmp_ptr, index=(c,), tile=dot_val)
    ct.store(scaled_dot_ptr, index=(c,), tile=scaled)


@ct.kernel
def _epilogue_kernel(
    x_ptr,        # bf16 (TOTAL,) contiguous
    mean_ptr,     # f32 (C,)
    invstd_ptr,   # f32 (C,)
    weight_ptr,   # f32 (C,)
    producer_ptr, # bf16 (TOTAL,) contiguous
    sum_ptr,      # f32 (C,)
    dot_ptr,      # f32 (C,)
    out_ptr,      # bf16 (TOTAL,) contiguous
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    producer = ct.load(producer_ptr, index=(pid,), shape=(BLOCK,))
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    invstd = ct.gather(invstd_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    sum_v = ct.gather(sum_ptr, channel)
    dot_v = ct.gather(dot_ptr, channel)

    x_f = ct.astype(x, ct.float32)
    producer_f = ct.astype(producer, ct.float32)
    centered = x_f - mean
    mean_term = sum_v * REDUCE_SCALE
    dot_scaled = dot_v * REDUCE_SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight
    after_variance = producer_f - centered * variance_term
    after_mean = after_variance - mean_term
    out_v = after_mean * output_scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out_v, ct.bfloat16))


@oracle_impl(hardware="B200", point="d723c5b4", GROUP_K=256, BLOCK_K=256, BLOCK_C=16)
def oracle_forward(inputs, *, GROUP_K: int, BLOCK_K: int, BLOCK_C: int):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs

    # Slice upstream = arg0[:, 160:320] as a contiguous NHWC (K_TOTAL, C) view.
    upstream_nhwc = arg0.permute(0, 2, 3, 1).contiguous()  # (N, H, W, 320)
    upstream_flat = upstream_nhwc[..., 160:320].contiguous().view(K_TOTAL, C)
    x_flat = arg1.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, C)

    mean = arg2.view(C)
    invstd = arg3.view(C)
    weight = arg4
    bias = arg5

    device = arg0.device
    num_k_tiles = (K_TOTAL + GROUP_K - 1) // GROUP_K
    block_tiles = _next_power_of_2(num_k_tiles)

    producer = torch.empty((K_TOTAL, C), device=device, dtype=torch.bfloat16)
    partial_sum = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C // BLOCK_C, num_k_tiles, 1),
        _partial_reduce_kernel,
        (upstream_flat, x_flat, mean, invstd, weight, bias,
         producer, partial_sum, partial_dot, GROUP_K, BLOCK_C),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd, sum_out, dot_tmp, scaled_dot,
         block_tiles),
    )

    BLOCK = 512
    out_flat = torch.empty((TOTAL,), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        ((TOTAL + BLOCK - 1) // BLOCK, 1, 1),
        _epilogue_kernel,
        (x_flat.view(-1), mean, invstd, weight, producer.view(-1),
         sum_out, dot_tmp, out_flat, C, BLOCK),
    )
    out = out_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    return sum_out, scaled_dot, out
