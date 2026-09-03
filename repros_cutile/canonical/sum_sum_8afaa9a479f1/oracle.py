"""cuTile port of sum_sum_8afaa9a479f1: MobileViT SiLU-backward + BN-backward.

The Triton oracle wraps every f32 op in `add.rn.f32/sub.rn.f32/mul.rn.f32`
inline PTX. cuTile is round-to-nearest-even by default, so we use `+`, `-`,
`*` directly.

Three-kernel pipeline:
  1. `_partial_reduce_kernel`: per-tile SiLU-backward producer + centered
     values, materializes producer_bf16 for reuse in the epilogue,
     and emits per-(c, tile) partial sums.
  2. `_finalize_kernel`: sums the per-channel partials, publishes sum_out and
     scaled_dot outputs.
  3. `_epilogue_kernel`: reads producer and finalized sums, computes the
     BN-backward tensor gradient.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 128
IN_C = 256
H = 16
W = 16
HW = H * W
K_TOTAL = N * HW  # 32768
TOTAL = K_TOTAL * C  # 4194304
REDUCE_SCALE = 3.0517578125e-05


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    grad_pair_ptr,   # bf16 [K_TOTAL, 256]  (grad_pair with arg0 layout)
    x_ptr,           # bf16 [K_TOTAL, 128]
    mean_ptr,        # f32 [128]
    invstd_ptr,      # f32 [128]
    weight_ptr,      # f32 [128]
    bias_ptr,        # f32 [128]
    producer_ptr,    # bf16 [K_TOTAL, 128]
    partial_sum_ptr, # f32 [128, num_k_tiles]  (c-major)
    partial_dot_ptr, # f32 [128, num_k_tiles]  (c-major)
    NUM_K_TILES: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    # Load per-channel scalars for the entire channel column (BLOCK_C).
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))

    # Load the (GROUP_K, BLOCK_C) tile of x and the sliced upstream grad.
    # grad_pair layout is [K_TOTAL, 256]; the slice arg0[:, 128:256] is
    # the second half of the second axis, i.e. col_block c_block + 1.
    # Since BLOCK_C=128 and total cols=256, the second half is c_block=1
    # when BLOCK_C=128. But if BLOCK_C < 128 we need to offset differently.
    # Easier: pass a pre-sliced tensor for grad_pair.

    upstream = ct.load(grad_pair_ptr, index=(k_block, c_block),
                       shape=(GROUP_K, BLOCK_C))
    x = ct.load(x_ptr, index=(k_block, c_block), shape=(GROUP_K, BLOCK_C))

    mean_row = ct.reshape(mean, (1, BLOCK_C))
    invstd_row = ct.reshape(invstd, (1, BLOCK_C))
    weight_row = ct.reshape(weight, (1, BLOCK_C))
    bias_row = ct.reshape(bias, (1, BLOCK_C))

    upstream_f = ct.astype(upstream, ct.float32)
    x_f = ct.astype(x, ct.float32)
    centered = x_f - mean_row
    normalized = centered * invstd_row
    affine_scaled = normalized * weight_row
    affine = affine_scaled + bias_row
    rounded_affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    exp_neg = ct.exp(0.0 - rounded_affine)
    sigmoid = 1.0 / (exp_neg + 1.0)
    grad_sigmoid = upstream_f * sigmoid
    one_minus = 1.0 - sigmoid
    affine_tail = rounded_affine * one_minus + 1.0
    producer_bf16 = ct.astype(grad_sigmoid * affine_tail, ct.bfloat16)
    producer_f = ct.astype(producer_bf16, ct.float32)

    ct.store(producer_ptr, index=(k_block, c_block), tile=producer_bf16)

    # per-channel partial sums along k
    sum_partial = ct.sum(producer_f, axis=0)  # (BLOCK_C,)
    dot_partial = ct.sum(producer_f * centered, axis=0)  # (BLOCK_C,)

    # partial layout: [C, NUM_K_TILES] with c-major so cols index -> k_tile.
    # Reshape (BLOCK_C,) -> (BLOCK_C, 1) so we can store into that slot.
    sum_2d = ct.reshape(sum_partial, (BLOCK_C, 1))
    dot_2d = ct.reshape(dot_partial, (BLOCK_C, 1))
    ct.store(partial_sum_ptr, index=(c_block, k_block), tile=sum_2d)
    ct.store(partial_dot_ptr, index=(c_block, k_block), tile=dot_2d)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [C, NUM_K_TILES]
    partial_dot_ptr,   # f32 [C, NUM_K_TILES]
    invstd_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    dot_tmp_ptr,       # f32 [C]
    scaled_dot_ptr,    # f32 [C]
    NUM_K_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    parts_sum = ct.load(partial_sum_ptr, index=(c, 0),
                        shape=(1, BLOCK_TILES),
                        padding_mode=ct.PaddingMode.ZERO)
    parts_dot = ct.load(partial_dot_ptr, index=(c, 0),
                        shape=(1, BLOCK_TILES),
                        padding_mode=ct.PaddingMode.ZERO)
    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_mask = ct.reshape(tiles < NUM_K_TILES, (1, BLOCK_TILES))
    parts_sum = ct.where(tile_mask, parts_sum, 0.0)
    parts_dot = ct.where(tile_mask, parts_dot, 0.0)

    sum_val = ct.sum(parts_sum)
    dot_val = ct.sum(parts_dot)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_val, (1,)))
    ct.store(dot_tmp_ptr, index=(c,), tile=ct.reshape(dot_val, (1,)))
    # invstd is a 1-element tile; sum reduces it to a scalar for multiplication.
    scaled = dot_val * ct.sum(invstd)
    ct.store(scaled_dot_ptr, index=(c,), tile=ct.reshape(scaled, (1,)))


@ct.kernel
def _epilogue_kernel(
    x_ptr,           # bf16 [K_TOTAL, C]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    producer_ptr,    # bf16 [K_TOTAL, C]
    sum_ptr,         # f32 [C]
    dot_ptr,         # f32 [C]
    out_ptr,         # bf16 [K_TOTAL, C]
    SCALE_C: ct.Constant[float],
    GROUP_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    k_block = ct.bid(0)
    c_block = ct.bid(1)
    x = ct.load(x_ptr, index=(k_block, c_block), shape=(GROUP_K, BLOCK_C))
    producer = ct.load(producer_ptr, index=(k_block, c_block),
                       shape=(GROUP_K, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    sum_v = ct.load(sum_ptr, index=(c_block,), shape=(BLOCK_C,))
    dot_v = ct.load(dot_ptr, index=(c_block,), shape=(BLOCK_C,))

    x_f = ct.astype(x, ct.float32)
    producer_f = ct.astype(producer, ct.float32)
    mean_row = ct.reshape(mean, (1, BLOCK_C))
    invstd_row = ct.reshape(invstd, (1, BLOCK_C))
    weight_row = ct.reshape(weight, (1, BLOCK_C))
    sum_row = ct.reshape(sum_v, (1, BLOCK_C))
    dot_row = ct.reshape(dot_v, (1, BLOCK_C))

    centered = x_f - mean_row
    mean_term = sum_row * SCALE_C
    dot_scaled = dot_row * SCALE_C
    variance_term = dot_scaled * (invstd_row * invstd_row)
    output_scale = invstd_row * weight_row
    after_variance = producer_f - centered * variance_term
    after_mean = after_variance - mean_term
    out = ct.astype(after_mean * output_scale, ct.bfloat16)
    ct.store(out_ptr, index=(k_block, c_block), tile=out)


@oracle_impl(
    hardware="B200",
    point="e23458e3",
    GROUP_K=256,
    BLOCK_C=128,
)
def oracle_forward(inputs, *, GROUP_K: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg1_1.device
    num_k_tiles = K_TOTAL // GROUP_K
    assert K_TOTAL % GROUP_K == 0

    # Views: channels-last [128,128,16,16] has stride [32768, 1, 2048, 128].
    # Reshaping to [K_TOTAL, C] with row-major (K_TOTAL, C=128, stride C, 1)
    # aligns with the channels-last layout: element (n, c, h, w) at
    # n*32768 + c*1 + h*2048 + w*128. In [K_TOTAL, C] indexing we want
    # k=(n*HW + h*W + w), c=c so element at k*C + c.
    # n*HW*C + h*W*C + w*C + c = n*(HW*C) + h*(W*C) + w*C + c.
    # Compare stride: (HW*C=32768, W*C=2048, C=128, 1) - matches strides for
    # channels-last [N, H, W, C]. So x.permute(0,2,3,1).contiguous().
    # But arg1_1 already has channels-last stride equivalent to that view.

    x = arg1_1.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, C)
    grad_pair = arg0_1.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, IN_C)
    # arg0[:, 128:256, :, :] = grad_pair[:, 128:256] in the reshaped layout.
    grad_upstream = grad_pair[:, C:].contiguous()  # [K_TOTAL, 128]

    mean_v = arg2_1.view(C).contiguous()
    invstd_v = arg3_1.view(C).contiguous()
    weight_v = arg4_1.contiguous()
    bias_v = arg5_1.contiguous()

    producer = torch.empty((K_TOTAL, C), device=device, dtype=torch.bfloat16)
    partial_sum = torch.empty((C, num_k_tiles), device=device, dtype=torch.float32)
    partial_dot = torch.empty((C, num_k_tiles), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((C,), device=device, dtype=torch.float32)
    out_flat = torch.empty((K_TOTAL, C), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C // BLOCK_C, num_k_tiles, 1),
        _partial_reduce_kernel,
        (grad_upstream, x, mean_v, invstd_v, weight_v, bias_v, producer,
         partial_sum, partial_dot, num_k_tiles, GROUP_K, BLOCK_C),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_v, sum_out, dot_tmp, scaled_dot,
         num_k_tiles, _next_power_of_2(num_k_tiles)),
    )
    ct.launch(
        stream,
        (num_k_tiles, C // BLOCK_C, 1),
        _epilogue_kernel,
        (x, mean_v, invstd_v, weight_v, producer, sum_out, dot_tmp, out_flat,
         REDUCE_SCALE, GROUP_K, BLOCK_C),
    )

    # Reshape out_flat [K_TOTAL, C] -> channels-last [N, C, H, W]
    out_channels_last = out_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last
    )

    return sum_out, scaled_dot, out_channels_last
