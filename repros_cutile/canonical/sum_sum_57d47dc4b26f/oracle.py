"""cuTile port of sum_sum_57d47dc4b26f: MobileViT SiLU+BN backward.

Structure mirrors the Triton reference verbatim:
1. `_partial_reduce_kernel` — 2D grid `(cdiv(C, BLOCK_C), num_k_tiles)` that
   materializes the bf16 SiLU-BN producer tile of shape `(BLOCK_K, BLOCK_C)`
   and emits per-channel partial `sum(producer)` and `sum(producer*centered)`
   into `partial_sum[c, k_tile]` and `partial_dot[c, k_tile]`.
2. `_finalize_kernel` — 1D grid `(C,)`. Each program reduces the two
   `(NUM_K_TILES,)` partial vectors for its channel into `sum_out[c]`,
   `dot_tmp[c]`, and `scaled_dot[c] = dot_value * invstd`.
3. `_epilogue_kernel` — 1D grid `(cdiv(TOTAL, EPILOGUE_BLOCK),)` running the
   BN-backward writeback: `after_variance = producer - centered * variance_term`,
   `after_mean = after_variance - mean_term`, `out = after_mean * output_scale`.

cuTile default round-to-nearest-even matches Triton's inline PTX rn intrinsics
and `.to(bf16, fp_downcast_rounding='rtne')` casts, so the numerics track.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 96
IN_C = 192
SLICE_OFFSET = 96
H = 32
W = 32
HW = H * W
K_TOTAL = N * HW           # 131072
TOTAL = K_TOTAL * C         # 12582912
REDUCE_SCALE = 7.62939453125e-06  # 1 / (N*H*W)


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    grad_pair_ptr,     # bf16 [K_TOTAL, C] view into arg0_1 at slice offset
    x_ptr,             # bf16 [K_TOTAL, C] flat view of arg1_1
    mean_ptr,          # f32  [C]
    invstd_ptr,        # f32  [C]
    weight_ptr,        # f32  [C]
    bias_ptr,          # f32  [C]
    producer_ptr,      # bf16 [K_TOTAL, C] flat writeback
    partial_sum_ptr,   # f32  [C, NUM_K_TILES]
    partial_dot_ptr,   # f32  [C, NUM_K_TILES]
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_tile = ct.bid(1)

    upstream = ct.load(grad_pair_ptr, index=(k_tile, c_block), shape=(BLOCK_K, BLOCK_C))
    x_bf = ct.load(x_ptr, index=(k_tile, c_block), shape=(BLOCK_K, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    upstream_f = ct.astype(upstream, ct.float32)
    x_f = ct.astype(x_bf, ct.float32)
    centered = x_f - mean_2d
    normalized = centered * invstd_2d
    affine = normalized * weight_2d + bias_2d
    rounded_affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    exp_neg = ct.exp(-rounded_affine)
    sigmoid = 1.0 / (exp_neg + 1.0)
    one_minus = 1.0 - sigmoid
    tail = rounded_affine * one_minus + 1.0
    producer_f = upstream_f * sigmoid * tail
    producer_bf = ct.astype(producer_f, ct.bfloat16)

    ct.store(producer_ptr, index=(k_tile, c_block), tile=producer_bf)

    producer_round = ct.astype(producer_bf, ct.float32)
    dot_tile = producer_round * centered
    partial_sum = ct.sum(producer_round, axis=0)   # (BLOCK_C,)
    partial_dot = ct.sum(dot_tile, axis=0)          # (BLOCK_C,)

    # partial_sum[c_block*BLOCK_C : (c_block+1)*BLOCK_C, k_tile]
    ct.store(partial_sum_ptr, index=(c_block, k_tile),
             tile=ct.reshape(partial_sum, (BLOCK_C, 1)))
    ct.store(partial_dot_ptr, index=(c_block, k_tile),
             tile=ct.reshape(partial_dot, (BLOCK_C, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 [C, NUM_K_TILES]
    partial_dot_ptr,     # f32 [C, NUM_K_TILES]
    invstd_ptr,          # f32 [C]
    sum_out_ptr,         # f32 [C]
    dot_tmp_ptr,         # f32 [C]
    scaled_dot_out_ptr,  # f32 [C]
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(c, 0), shape=(1, BLOCK_TILES))
    p_dot = ct.load(partial_dot_ptr, index=(c, 0), shape=(1, BLOCK_TILES))

    sum_value = ct.sum(p_sum, axis=1)   # (1,)
    dot_value = ct.sum(p_dot, axis=1)   # (1,)
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    scaled = dot_value * invstd

    ct.store(sum_out_ptr, index=(c,), tile=sum_value)
    ct.store(dot_tmp_ptr, index=(c,), tile=dot_value)
    ct.store(scaled_dot_out_ptr, index=(c,), tile=scaled)


@ct.kernel
def _epilogue_kernel(
    x_ptr,           # bf16 [TOTAL]
    mean_ptr,        # f32  [C]
    invstd_ptr,      # f32  [C]
    weight_ptr,      # f32  [C]
    producer_ptr,    # bf16 [TOTAL]
    sum_ptr,         # f32  [C]
    dot_ptr,         # f32  [C]
    out_ptr,         # bf16 [TOTAL]
    C_: ct.Constant[int],
    REDUCE_SCALE_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lanes = ct.arange(BLOCK, dtype=ct.int32)
    offsets = pid * BLOCK + lanes
    c_idx = offsets % C_

    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    producer_bf = ct.load(producer_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x_bf, ct.float32)
    producer_f = ct.astype(producer_bf, ct.float32)

    mean = ct.gather(mean_ptr, c_idx)
    invstd = ct.gather(invstd_ptr, c_idx)
    weight = ct.gather(weight_ptr, c_idx)
    sum_value = ct.gather(sum_ptr, c_idx)
    dot_value = ct.gather(dot_ptr, c_idx)

    centered = x_f - mean
    mean_term = sum_value * REDUCE_SCALE_
    dot_scaled = dot_value * REDUCE_SCALE_
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight
    after_variance = producer_f - centered * variance_term
    after_mean = after_variance - mean_term
    out_f = after_mean * output_scale

    ct.store(out_ptr, index=(pid,), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(
    hardware="B200",
    point="65b58dc6",
    GROUP_K=256,
    BLOCK_K=256,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
)
def oracle_forward(
    inputs,
    *,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
):
    del GROUP_K  # GROUP_K == BLOCK_K in this schedule; kept for signature parity
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg1_1.device

    num_k_tiles = K_TOTAL // BLOCK_K
    block_tiles = _next_pow2(num_k_tiles)

    # 1D per-channel BN parameters.
    mean_1d = arg2_1.view(C).contiguous()
    invstd_1d = arg3_1.view(C).contiguous()
    weight_1d = arg4_1.contiguous()
    bias_1d = arg5_1.contiguous()

    # (K_TOTAL, C) 2D views into the channels-last packed storage. arg0_1
    # has 192 packed channels; the slice starts at storage_offset=SLICE_OFFSET
    # with row stride IN_C=192. arg1_1 has C=96 packed channels, stride C.
    grad_pair_2d = torch.as_strided(
        arg0_1, (K_TOTAL, C), (IN_C, 1), storage_offset=SLICE_OFFSET
    )
    x_2d = torch.as_strided(arg1_1, (K_TOTAL, C), (C, 1))

    # Outputs and scratch. producer_2d and out_2d share the channels-last
    # storage layout with arg1_1 so their (K_TOTAL, C) view is dense.
    producer_cl = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    producer_2d = torch.as_strided(producer_cl, (K_TOTAL, C), (C, 1))
    out_cl = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    producer_flat = torch.as_strided(producer_cl, (TOTAL,), (1,))
    x_flat = torch.as_strided(arg1_1, (TOTAL,), (1,))
    out_flat = torch.as_strided(out_cl, (TOTAL,), (1,))

    partial_sum = torch.empty(
        (C, num_k_tiles), device=device, dtype=torch.float32,
    )
    partial_dot = torch.empty(
        (C, num_k_tiles), device=device, dtype=torch.float32,
    )
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (C // BLOCK_C, num_k_tiles, 1),
        _partial_reduce_kernel,
        (
            grad_pair_2d, x_2d, mean_1d, invstd_1d, weight_1d, bias_1d,
            producer_2d, partial_sum, partial_dot,
            BLOCK_K, BLOCK_C,
        ),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (
            partial_sum, partial_dot, invstd_1d,
            sum_out, dot_tmp, scaled_dot,
            block_tiles,
        ),
    )
    ct.launch(
        stream,
        (TOTAL // EPILOGUE_BLOCK, 1, 1),
        _epilogue_kernel,
        (
            x_flat, mean_1d, invstd_1d, weight_1d, producer_flat,
            sum_out, dot_tmp, out_flat,
            C, REDUCE_SCALE, EPILOGUE_BLOCK,
        ),
    )

    return sum_out, scaled_dot, out_cl
