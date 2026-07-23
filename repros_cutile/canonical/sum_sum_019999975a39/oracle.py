"""cuTile port of sum_sum_019999975a39: NHWC BN-backward avg-pool-backward.

Three-kernel pipeline mirroring the Triton oracle:
1. Partial-reduction kernel: computes per-tile sums (sum(grad_silu) and
   sum(grad_silu * centered)) into partial buffers.
2. Finalize kernel: reduces partials into per-channel scalars and computes
   sum_1 and scaled_dot.
3. Epilogue kernel: recomputes grad_silu and writes the final bf16 output.

inline PTX add/sub/mul/div rn are cuTile defaults so drop to plain ops.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _partial_reduce_kernel(
    pooled,           # bf16 (N, C)
    x,                # bf16 (K_TOTAL, C)  (channels-last flatten)
    mean,             # f32 (C,)
    invstd,           # f32 (C,)
    weight,           # f32 (C,)
    bias,             # f32 (C,)
    partial_sum,      # f32 (C, NUM_K_TILES)
    partial_dot,      # f32 (C, NUM_K_TILES)
    C: ct.Constant[int],
    SPATIAL: ct.Constant[int],
    DIVISOR: ct.Constant[float],
    K_TOTAL: ct.Constant[int],
    NUM_K_TILES: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    tile_idx = ct.bid(1)

    c = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    k_lanes = ct.arange(BLOCK_K, dtype=ct.int32)
    k = tile_idx * GROUP_K + k_lanes
    n = k // SPATIAL
    c_valid = c < C
    k_valid = (k_lanes < GROUP_K) & (k < K_TOTAL)
    # (BLOCK_K, BLOCK_C)
    k2 = ct.reshape(k, (BLOCK_K, 1))
    n2 = ct.reshape(n, (BLOCK_K, 1))
    c2 = ct.reshape(c, (1, BLOCK_C))
    k_valid2 = ct.reshape(k_valid, (BLOCK_K, 1))
    c_valid2 = ct.reshape(c_valid, (1, BLOCK_C))
    mask2 = k_valid2 & c_valid2

    k_b = ct.broadcast_to(k2, (BLOCK_K, BLOCK_C))
    n_b = ct.broadcast_to(n2, (BLOCK_K, BLOCK_C))
    c_b = ct.broadcast_to(c2, (BLOCK_K, BLOCK_C))

    x_bf = ct.gather(x, (k_b, c_b), mask=mask2)
    x_f = ct.astype(x_bf, ct.float32)
    pooled_bf = ct.gather(pooled, (n_b, c_b), mask=mask2)
    pooled_f = ct.astype(pooled_bf, ct.float32)

    mean_val = ct.gather(mean, c, mask=c_valid)
    invstd_val = ct.gather(invstd, c, mask=c_valid)
    weight_val = ct.gather(weight, c, mask=c_valid)
    bias_val = ct.gather(bias, c, mask=c_valid)

    pool_grad = ct.astype(ct.astype(pooled_f / DIVISOR, ct.bfloat16), ct.float32)
    centered = x_f - ct.reshape(mean_val, (1, BLOCK_C))
    affine = centered * ct.reshape(invstd_val, (1, BLOCK_C)) \
        * ct.reshape(weight_val, (1, BLOCK_C)) \
        + ct.reshape(bias_val, (1, BLOCK_C))
    affine_bf = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    sigmoid = 1.0 / (1.0 + ct.exp(-affine_bf))
    grad_silu = pool_grad * sigmoid * (affine_bf * (1.0 - sigmoid) + 1.0)
    grad_silu = ct.astype(ct.astype(grad_silu, ct.bfloat16), ct.float32)

    active = ct.where(mask2, grad_silu, 0.0)
    dot = ct.where(mask2, grad_silu * centered, 0.0)
    partial_s = ct.sum(active, axis=0)
    partial_d = ct.sum(dot, axis=0)

    # Store partial[c, tile_idx] for each c in BLOCK_C.
    tile_b = ct.full((BLOCK_C,), tile_idx, dtype=ct.int32)
    ct.scatter(partial_sum, (c, tile_b), partial_s, mask=c_valid)
    ct.scatter(partial_dot, (c, tile_b), partial_d, mask=c_valid)


@ct.kernel
def _finalize_kernel(
    partial_sum,   # f32 (C, NUM_K_TILES)
    partial_dot,   # f32 (C, NUM_K_TILES)
    invstd,        # f32 (C,)
    sum_out,       # f32 (C,)
    scaled_dot_out,  # f32 (C,)
    dot_tmp,       # f32 (C,)
    NUM_K_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_valid = tiles < NUM_K_TILES
    c_tile = ct.full((BLOCK_TILES,), c, dtype=ct.int32)
    sum_vals = ct.gather(partial_sum, (c_tile, tiles), mask=tile_valid,
                         padding_value=0)
    dot_vals = ct.gather(partial_dot, (c_tile, tiles), mask=tile_valid,
                         padding_value=0)
    sum_vals = ct.where(tile_valid, sum_vals, 0.0)
    dot_vals = ct.where(tile_valid, dot_vals, 0.0)
    sum_value = ct.sum(sum_vals)
    dot_value = ct.sum(dot_vals)
    inv = ct.load(invstd, index=(c,), shape=(1,))
    inv_f = ct.reshape(ct.astype(inv, ct.float32), ())
    ct.store(sum_out, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(dot_tmp, index=(c,), tile=ct.reshape(dot_value, (1,)))
    ct.store(scaled_dot_out, index=(c,),
             tile=ct.reshape(dot_value * inv_f, (1,)))


@ct.kernel
def _epilogue_kernel(
    pooled,        # bf16 (N, C)
    x,             # bf16 (K_TOTAL, C)
    mean,          # f32 (C,)
    invstd,        # f32 (C,)
    weight,        # f32 (C,)
    bias,          # f32 (C,)
    sum_v,         # f32 (C,)
    dot_v,         # f32 (C,)
    grad_input,    # bf16 (K_TOTAL, C)
    TOTAL: ct.Constant[int],
    C: ct.Constant[int],
    SPATIAL: ct.Constant[int],
    DIVISOR: ct.Constant[float],
    REDUCE_SCALE: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = offsets < TOTAL
    c_idx = offsets % C
    k_idx = offsets // C
    n_idx = k_idx // SPATIAL
    c_safe = ct.where(valid, c_idx, 0)

    # Load x[k, c]
    x_bf = ct.gather(x, (k_idx, c_idx), mask=valid)
    x_f = ct.astype(x_bf, ct.float32)
    pooled_bf = ct.gather(pooled, (n_idx, c_idx), mask=valid)
    pooled_f = ct.astype(pooled_bf, ct.float32)

    mean_val = ct.gather(mean, c_safe)
    invstd_val = ct.gather(invstd, c_safe)
    weight_val = ct.gather(weight, c_safe)
    bias_val = ct.gather(bias, c_safe)
    sum_val = ct.gather(sum_v, c_safe)
    dot_val = ct.gather(dot_v, c_safe)

    pool_grad = ct.astype(ct.astype(pooled_f / DIVISOR, ct.bfloat16), ct.float32)
    centered = x_f - mean_val
    affine = centered * invstd_val * weight_val + bias_val
    affine_bf = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    sigmoid = 1.0 / (1.0 + ct.exp(-affine_bf))
    grad_silu = pool_grad * sigmoid * (affine_bf * (1.0 - sigmoid) + 1.0)
    grad_silu = ct.astype(ct.astype(grad_silu, ct.bfloat16), ct.float32)

    mean_term = sum_val * REDUCE_SCALE
    var_term = dot_val * REDUCE_SCALE * invstd_val * invstd_val
    input_scale = invstd_val * weight_val
    correction = grad_silu - centered * var_term - mean_term
    grad_in = ct.astype(correction * input_scale, ct.bfloat16)
    ct.scatter(grad_input, (k_idx, c_idx), grad_in, mask=valid)


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


def _run_oracle(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
):
    pooled, x, mean, invstd, weight, bias = inputs[:6]
    device = x.device
    spatial = H * W
    k_total = 128 * spatial
    total = k_total * C

    # Flatten pooled (128, C, 1, 1) → (128, C).
    pooled_2d = pooled.view(128, C)

    # x is channels-last (128, C, H, W) with stride like (C*H*W, 1, W*C, C).
    # In channels-last, iterating linearly over storage gives (n, h, w, c).
    # So x.reshape(k_total, C) via as_strided on storage gives (K_TOTAL, C)
    # with row = n*H*W + h*W + w.
    x_2d = torch.as_strided(x, (k_total, C), (C, 1))

    # Output has same strides as x.
    grad_input = torch.empty_strided(tuple(x.shape), tuple(x.stride()),
                                     device=device, dtype=torch.bfloat16)
    grad_input_2d = torch.as_strided(grad_input, (k_total, C), (C, 1))

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((C,), device=device, dtype=torch.float32)

    num_k_tiles = (k_total + GROUP_K - 1) // GROUP_K
    partial_sum = torch.empty((C, num_k_tiles), device=device, dtype=torch.float32)
    partial_dot = torch.empty((C, num_k_tiles), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((C + BLOCK_C - 1) // BLOCK_C, num_k_tiles, 1),
        _partial_reduce_kernel,
        (pooled_2d, x_2d, mean.view(C), invstd.view(C), weight, bias,
         partial_sum, partial_dot, C, spatial, 64.0, k_total, num_k_tiles,
         GROUP_K, BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd.view(C), sum_out, scaled_dot_out,
         dot_tmp, num_k_tiles, _next_pow2(num_k_tiles)),
    )
    ct.launch(
        stream,
        ((total + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK, 1, 1),
        _epilogue_kernel,
        (pooled_2d, x_2d, mean.view(C), invstd.view(C), weight, bias,
         sum_out, dot_tmp, grad_input_2d, total, C, spatial, 64.0,
         0.0001220703125, EPILOGUE_BLOCK),
    )
    return sum_out, scaled_dot_out, grad_input


@oracle_impl(hardware="B200", point="79fb3aff", C=640, H=8, W=8, GROUP_K=256, BLOCK_K=256, BLOCK_C=16, EPILOGUE_BLOCK=256)
@oracle_impl(hardware="B200", point="a44186d7", C=1280, H=7, W=7, GROUP_K=256, BLOCK_K=256, BLOCK_C=16, EPILOGUE_BLOCK=256)
def oracle_forward(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
):
    return _run_oracle(
        inputs,
        C=C, H=H, W=W,
        GROUP_K=GROUP_K, BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
    )
