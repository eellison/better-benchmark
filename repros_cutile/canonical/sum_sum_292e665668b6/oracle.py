"""cuTile port of sum_sum_292e665668b6: EfficientNet SE-gate + BN-backward.

Matches Triton's 3-kernel structure:
  1. `_partial_reduce_kernel` — computes the SE-gate producer (bf16 add/div
     + hard-swish-derivative gate) for the full tile, writes it to a f32
     producer buffer, and accumulates partial sums `sum_1` and
     `sum_2 = sum(producer * (arg5 - mean))` per channel.
  2. `_finalize_kernel` — reduces partials to per-channel `sum_1`, `dot`,
     and derived `mul_12 = dot * invstd`.
  3. `_epilogue_kernel` — per-element BN-backward output over producer.

All reductions, sigmoid, hard-swish-derivative math live inside the cuTile
kernels; torch does only allocation and layout.

The Triton reference uses overlapping tiles (GROUP_K < BLOCK_K with mask);
cuTile uses non-overlapping BLOCK_K-stride tiles for the same partition, so
NUM_K_TILES here = cdiv(K_TOTAL, BLOCK_K), not cdiv(K_TOTAL, GROUP_K).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 6.228077168367346e-07
EPILOGUE_BLOCK = 1024


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    arg0_ptr,        # bf16 [K_TOTAL, C] channels-last flat
    arg1_ptr,        # bf16 [K_TOTAL, C] broadcast-expanded
    arg2_ptr,        # bf16 [K_TOTAL, C] broadcast-expanded
    gate_ptr,        # bf16 [K_TOTAL, C] channels-last flat (arg3)
    mean_input_ptr,  # bf16 [K_TOTAL, C] channels-last flat (arg5)
    mean_ptr,        # f32  [C]
    producer_ptr,    # f32  [K_TOTAL, C]
    partial_sum_ptr, # f32  [NUM_K_TILES, C]
    partial_dot_ptr, # f32  [NUM_K_TILES, C]
    C_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_tile = ct.bid(1)

    # Load arg0/arg1/arg2/gate/mean_input tiles.
    arg0 = ct.load(arg0_ptr, index=(k_tile, c_block), shape=(BLOCK_K, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    arg1 = ct.load(arg1_ptr, index=(k_tile, c_block), shape=(BLOCK_K, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    arg2 = ct.load(arg2_ptr, index=(k_tile, c_block), shape=(BLOCK_K, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    gate_bf = ct.load(gate_ptr, index=(k_tile, c_block), shape=(BLOCK_K, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    mean_input_bf = ct.load(mean_input_ptr, index=(k_tile, c_block),
                            shape=(BLOCK_K, BLOCK_C),
                            padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))

    a0_f = ct.astype(arg0, ct.float32)
    a1_f = ct.astype(arg1, ct.float32)
    a2_f = ct.astype(arg2, ct.float32)
    gate_f = ct.astype(gate_bf, ct.float32)
    mean_input_f = ct.astype(mean_input_bf, ct.float32)

    # bf16 round-trip mul: mul_term = bf16(a0 * a1) -> f32
    mul_term = ct.astype(ct.astype(a0_f * a1_f, ct.bfloat16), ct.float32)
    # div_term = bf16(a2 / 12544) -> f32
    div_term = ct.astype(ct.astype(a2_f * (1.0 / 12544.0), ct.bfloat16), ct.float32)
    # add_term = bf16(mul + div) -> f32
    add_term = ct.astype(ct.astype(mul_term + div_term, ct.bfloat16), ct.float32)

    # sigmoid(gate) via 1/(1+exp(-gate))
    sigmoid = 1.0 / (1.0 + ct.exp(-gate_f))
    first = add_term * sigmoid
    tail = gate_f * (1.0 - sigmoid) + 1.0
    producer_val = first * tail
    # Round-trip through bf16 (matches Triton's .to(tl.bfloat16).to(tl.float32))
    producer = ct.astype(ct.astype(producer_val, ct.bfloat16), ct.float32)

    # Store producer to buffer with the same tile-space mapping.
    ct.store(producer_ptr, index=(k_tile, c_block), tile=producer)

    # Row/column mask for tail tile.
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32) + k_tile * BLOCK_K
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    k_valid = ct.reshape(k_idx < K_TOTAL_, (BLOCK_K, 1))
    c_valid = ct.reshape(c_idx < C_, (1, BLOCK_C))
    valid = k_valid & c_valid

    centered = mean_input_f - mean_2d
    zero_f = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32)
    active = ct.where(valid, producer, zero_f)
    active_dot = ct.where(valid, producer * centered, zero_f)

    p_sum = ct.sum(active, axis=0, keepdims=True)   # (1, BLOCK_C)
    p_dot = ct.sum(active_dot, axis=0, keepdims=True)

    ct.store(partial_sum_ptr, index=(k_tile, c_block), tile=p_sum)
    ct.store(partial_dot_ptr, index=(k_tile, c_block), tile=p_dot)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [NUM_K_TILES, C]
    partial_dot_ptr,   # f32 [NUM_K_TILES, C]
    invstd_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    dot_out_ptr,       # f32 [C]
    scaled_dot_ptr,    # f32 [C] = dot * invstd
    C_: ct.Constant[int],
    NUM_K_TILES_: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    tile = ct.load(partial_sum_ptr, index=(0, c_block),
                   shape=(BLOCK_TILES, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    tile2 = ct.load(partial_dot_ptr, index=(0, c_block),
                    shape=(BLOCK_TILES, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    t_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    t_valid = ct.reshape(t_idx < NUM_K_TILES_, (BLOCK_TILES, 1))
    c_valid = ct.reshape(c_idx < C_, (1, BLOCK_C))
    valid = t_valid & c_valid
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
    tile = ct.where(valid, tile, zero_f)
    tile2 = ct.where(valid, tile2, zero_f)

    sum_value = ct.sum(tile, axis=0)
    dot_value = ct.sum(tile2, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    scaled = dot_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(dot_out_ptr, index=(c_block,), tile=dot_value)
    ct.store(scaled_dot_ptr, index=(c_block,), tile=scaled)


@ct.kernel
def _epilogue_kernel(
    mean_input_ptr,   # bf16 flat (TOTAL,)
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    producer_ptr,     # f32 flat (TOTAL,)
    sum_ptr,          # f32 [C]
    dot_ptr,          # f32 [C]
    out_ptr,          # bf16 flat (TOTAL,)
    TOTAL_: ct.Constant[int],
    C_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    active = offsets < TOTAL_
    c = offsets - (offsets // C_) * C_

    mean_input = ct.astype(
        ct.gather(mean_input_ptr, offsets, mask=active),
        ct.float32,
    )
    mean = ct.gather(mean_ptr, c, mask=active)
    invstd = ct.gather(invstd_ptr, c, mask=active)
    weight = ct.gather(weight_ptr, c, mask=active)
    producer = ct.gather(producer_ptr, offsets, mask=active)
    sum_value = ct.gather(sum_ptr, c, mask=active)
    dot_value = ct.gather(dot_ptr, c, mask=active)

    centered = mean_input - mean
    mean_term = sum_value * SCALE_
    dot_scaled = dot_value * SCALE_
    correction = dot_scaled * (invstd * invstd)
    output_scale = invstd * weight
    without_var = producer - centered * correction
    without_mean = without_var - mean_term
    result = without_mean * output_scale
    ct.scatter(out_ptr, offsets, ct.astype(result, ct.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="4cb33acf", GROUP_K=1568, BLOCK_K=2048, BLOCK_C=8)
@oracle_impl(hardware="B200", point="5a064106", GROUP_K=392, BLOCK_K=512, BLOCK_C=16)
@oracle_impl(hardware="B200", point="243c10e8", GROUP_K=392, BLOCK_K=512, BLOCK_C=16)
@oracle_impl(hardware="B200", point="0a23c223", GROUP_K=196, BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="48ea6f2c", GROUP_K=196, BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="631acb3f", GROUP_K=196, BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="1a6807fb", GROUP_K=196, BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="61f1c546", GROUP_K=196, BLOCK_K=256, BLOCK_C=16)
@oracle_impl(hardware="B200", point="3a1bd77b", GROUP_K=49, BLOCK_K=64, BLOCK_C=16)
@oracle_impl(hardware="B200", point="bf467b0a", GROUP_K=49, BLOCK_K=64, BLOCK_C=16)
def oracle_forward(inputs, *, GROUP_K: int, BLOCK_K: int, BLOCK_C: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, shape0,
    ) = inputs
    device = arg0_1.device
    n, c, h, w = arg0_1.shape
    hw = h * w
    k_total = n * hw
    total = n * c * hw

    # Use BLOCK_K as tile stride (cuTile doesn't do Triton's overlap-with-mask).
    num_k_tiles = (k_total + BLOCK_K - 1) // BLOCK_K
    block_tiles = _next_power_of_2(num_k_tiles)

    # Materialize channels-last as (K_TOTAL, C) contiguous.
    arg0_flat = arg0_1.permute(0, 2, 3, 1).contiguous().view(k_total, c)
    arg3_flat = arg3_1.permute(0, 2, 3, 1).contiguous().view(k_total, c)  # gate
    arg5_flat = arg5_1.permute(0, 2, 3, 1).contiguous().view(k_total, c)

    # arg1/arg2 are [N, C, 1, 1] -> expand to [K_TOTAL, C].
    arg1_expand = arg1_1.expand(n, c, h, w).permute(0, 2, 3, 1).contiguous().view(k_total, c)
    arg2_expand = arg2_1.expand(n, c, h, w).permute(0, 2, 3, 1).contiguous().view(k_total, c)

    mean_1d = arg4_1.view(c).contiguous()
    invstd_1d = arg6_1.view(c).contiguous()
    weight_1d = arg7_1.contiguous()

    producer_flat = torch.empty((k_total, c), device=device, dtype=torch.float32)
    partial_sum = torch.empty((num_k_tiles, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, c), device=device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    dot_out = torch.empty((c,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    num_c_blocks = (c + BLOCK_C - 1) // BLOCK_C

    ct.launch(
        stream,
        (num_c_blocks, num_k_tiles, 1),
        _partial_reduce_kernel,
        (arg0_flat, arg1_expand, arg2_expand, arg3_flat, arg5_flat, mean_1d,
         producer_flat, partial_sum, partial_dot,
         c, k_total, BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, sum_out, dot_out, scaled_dot,
         c, num_k_tiles, block_tiles, BLOCK_C),
    )

    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        ((total + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK, 1, 1),
        _epilogue_kernel,
        (arg5_flat.view(-1), mean_1d, invstd_1d, weight_1d,
         producer_flat.view(-1), sum_out, dot_out, out_flat,
         total, c, REDUCE_SCALE, EPILOGUE_BLOCK),
    )

    out = torch.empty_strided(
        tuple(int(d) for d in shape0),
        (c * h * w, 1, c * w, c),
        device=device, dtype=torch.bfloat16,
    )
    out.copy_(out_flat.view(n, h, w, c).permute(0, 3, 1, 2))

    return sum_out, scaled_dot, out
