"""cuTile port of sum_sum_808b052d7393: GhostNet BN-backward + hard-sigmoid gate.

Mirrors Triton's 3-kernel structure:
  1. partials_kernel: BN affine gate + adaptive-pool-backward `where`,
     computes per-channel partials `sum(selected)` and
     `sum(selected * centered)` via `ct.sum(...)`.
  2. finalize_kernel: reduces partials into per-channel scalars, precomputes
     coefficients.
  3. epilogue_kernel: applies coefficients to write bf16 out gradient.

Reductions live inside `@ct.kernel` via `ct.sum(...)` — no torch `.sum(...)`
in oracle_forward. BLOCK_R=128, BLOCK_C=16, BLOCK_E=256 match Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH  # 49
REDUCE_SIZE = BATCH * HW  # 25088
NUMEL = BATCH * CHANNELS * HW
SCALE = 3.985969387755102e-05


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partials_kernel(
    pooled_ptr,           # bf16 [BATCH, CHANNELS]
    activation_ptr,       # bf16 [REDUCE_SIZE, CHANNELS]  (NHWC-flat)
    mean_ptr,             # f32  [CHANNELS]
    invstd_ptr,           # f32  [CHANNELS]
    weight_ptr,           # f32  [CHANNELS]
    bias_ptr,             # f32  [CHANNELS]
    scalar_ptr,           # bf16 [1]
    partial_sum_ptr,      # f32  [num_tiles, CHANNELS]
    partial_centered_ptr, # f32  [num_tiles, CHANNELS]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    REDUCE_SIZE_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_block = ct.bid(1)

    row_idx = ct.arange(BLOCK_R, dtype=ct.int32) + tile * BLOCK_R
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    row_valid = ct.reshape(row_idx < REDUCE_SIZE_C, (BLOCK_R, 1))
    col_valid = ct.reshape(col_idx < C_C, (1, BLOCK_C))
    active = row_valid & col_valid

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    scalar_v = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_f = ct.astype(scalar_v, ct.float32)

    activation_bf = ct.load(activation_ptr, index=(tile, c_block),
                            shape=(BLOCK_R, BLOCK_C),
                            padding_mode=ct.PaddingMode.ZERO)
    activation = ct.astype(activation_bf, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    centered = activation - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_gate = ct.astype(affine_bf, ct.float32)
    take_pool = active & (affine_gate > 0.0)

    # pooled: shape (BATCH*CHANNELS,) flat — gather via computed offsets. For
    # tile-space, pooled index is (batch=row_idx // HW, col=col_idx).
    # Reduce_index = tile*BLOCK_R + i, batch = reduce_index // HW.
    batch_idx = row_idx // HW_C  # (BLOCK_R,)
    batch_offsets = ct.reshape(batch_idx, (BLOCK_R, 1)) * C_C + ct.reshape(col_idx, (1, BLOCK_C))
    pooled_bf = ct.gather(pooled_ptr, batch_offsets)  # pooled_ptr is 1-D flat
    pooled = ct.astype(pooled_bf, ct.float32)
    pool_bf16 = ct.astype(pooled * (1.0 / 49.0), ct.bfloat16)
    scalar_bcast = ct.broadcast_to(ct.astype(scalar_v, ct.bfloat16), (BLOCK_R, BLOCK_C))
    selected_bf = ct.where(take_pool, pool_bf16, scalar_bcast)
    selected = ct.astype(selected_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    selected_masked = ct.where(active, selected, zero_f)
    centered_masked = ct.where(active, centered, zero_f)

    p_sum = ct.sum(selected_masked, axis=0, keepdims=True)
    p_centered = ct.sum(selected_masked * centered_masked, axis=0, keepdims=True)
    ct.store(partial_sum_ptr, index=(tile, c_block), tile=p_sum)
    ct.store(partial_centered_ptr, index=(tile, c_block), tile=p_centered)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 [num_tiles, CHANNELS]
    partial_centered_ptr, # f32 [num_tiles, CHANNELS]
    invstd_ptr,           # f32 [CHANNELS]
    weight_ptr,           # f32 [CHANNELS]
    sum_out_ptr,          # f32 [CHANNELS]
    vector_out_ptr,       # f32 [CHANNELS]
    coeff_mean_ptr,       # f32 [CHANNELS]
    coeff_var_ptr,        # f32 [CHANNELS]
    coeff_weight_ptr,     # f32 [CHANNELS]
    C_C: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, c_block),
                    shape=(BLOCK_TILES, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_centered = ct.load(partial_centered_ptr, index=(0, c_block),
                         shape=(BLOCK_TILES, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
    t_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    t_valid = ct.reshape(t_idx < NUM_TILES, (BLOCK_TILES, 1))
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
    p_sum_v = ct.where(t_valid, p_sum, zero_f)
    p_centered_v = ct.where(t_valid, p_centered, zero_f)
    sum_where = ct.sum(p_sum_v, axis=0)
    sum_centered = ct.sum(p_centered_v, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)

    scaled_sum = sum_where * SCALE
    scaled_centered = sum_centered * SCALE
    invstd_sq = invstd * invstd
    coeff_var = scaled_centered * invstd_sq
    coeff_weight = invstd * weight
    vector_out = sum_centered * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_where)
    ct.store(vector_out_ptr, index=(c_block,), tile=vector_out)
    ct.store(coeff_mean_ptr, index=(c_block,), tile=scaled_sum)
    ct.store(coeff_var_ptr, index=(c_block,), tile=coeff_var)
    ct.store(coeff_weight_ptr, index=(c_block,), tile=coeff_weight)


@ct.kernel
def _epilogue_kernel(
    pooled_ptr,           # bf16 [NUMEL] NHWC-flat access via reduce_index // HW
    activation_ptr,       # bf16 [NUMEL] NHWC-flat
    mean_ptr,             # f32  [CHANNELS]
    invstd_ptr,           # f32  [CHANNELS]
    weight_ptr,           # f32  [CHANNELS]
    bias_ptr,             # f32  [CHANNELS]
    scalar_ptr,           # bf16 [1]
    coeff_mean_ptr,       # f32  [CHANNELS]
    coeff_var_ptr,        # f32  [CHANNELS]
    coeff_weight_ptr,     # f32  [CHANNELS]
    out_ptr,              # bf16 [NUMEL] NHWC-flat
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    linear = ct.arange(BLOCK_E, dtype=ct.int32) + pid * BLOCK_E
    c = linear - (linear // C_C) * C_C
    nhw = linear // C_C
    n = nhw // HW_C
    pooled_offsets = n * C_C + c

    mean = ct.gather(mean_ptr, c)
    invstd = ct.gather(invstd_ptr, c)
    weight = ct.gather(weight_ptr, c)
    bias = ct.gather(bias_ptr, c)
    scalar_v = ct.load(scalar_ptr, index=(0,), shape=(1,))

    activation_bf = ct.load(activation_ptr, index=(pid,), shape=(BLOCK_E,))
    activation = ct.astype(activation_bf, ct.float32)
    centered = activation - mean

    affine = centered * invstd * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_gate = ct.astype(affine_bf, ct.float32)
    take_pool = affine_gate > 0.0

    pooled_bf = ct.gather(pooled_ptr, pooled_offsets)
    pooled = ct.astype(pooled_bf, ct.float32)
    pool_bf16 = ct.astype(pooled * (1.0 / 49.0), ct.bfloat16)
    scalar_bcast = ct.broadcast_to(ct.astype(scalar_v, ct.bfloat16), (BLOCK_E,))
    selected_bf = ct.where(take_pool, pool_bf16, scalar_bcast)
    selected = ct.astype(selected_bf, ct.float32)

    coeff_mean = ct.gather(coeff_mean_ptr, c)
    coeff_var = ct.gather(coeff_var_ptr, c)
    coeff_weight = ct.gather(coeff_weight_ptr, c)
    variance_term = centered * coeff_var
    without_var = selected - variance_term
    without_mean = without_var - coeff_mean
    result = without_mean * coeff_weight
    ct.store(out_ptr, index=(pid,), tile=ct.astype(result, ct.bfloat16))


@oracle_impl(hardware="B200", point="69efee57", BLOCK_R=128, BLOCK_C=16, BLOCK_E=256)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C, BLOCK_E):
    (pooled, activation, mean, invstd, weight, bias, scalar,
     _s0, _s1, _s2, _s3, _s4, _s5) = inputs
    device = activation.device

    # pooled: bf16[512, 960, 1, 1] contiguous → flat (BATCH*CHANNELS,)
    pooled_flat = pooled.view(BATCH * CHANNELS)
    # activation: bf16[512, 960, 7, 7] channels-last (stride 47040,1,6720,960).
    # permute(0,2,3,1) → stride (47040,6720,960,1) which is already contiguous,
    # so no copy is needed.
    activation_flat_full = activation.permute(0, 2, 3, 1).reshape(-1)
    activation_2d = activation_flat_full.view(REDUCE_SIZE, CHANNELS)

    mean_flat = mean.view(CHANNELS)
    invstd_flat = invstd.view(CHANNELS)
    weight_flat = weight.view(CHANNELS)
    bias_flat = bias.view(CHANNELS)
    scalar_flat = scalar.view(1)

    num_tiles = (REDUCE_SIZE + BLOCK_R - 1) // BLOCK_R
    block_tiles = _next_power_of_2(num_tiles)
    num_c_blocks = (CHANNELS + BLOCK_C - 1) // BLOCK_C

    partial_sum = torch.empty((num_tiles, CHANNELS), device=device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    vector_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    coeff_mean = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    coeff_var = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    coeff_weight = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_tiles, num_c_blocks, 1),
        _partials_kernel,
        (pooled_flat, activation_2d, mean_flat, invstd_flat, weight_flat,
         bias_flat, scalar_flat, partial_sum, partial_centered,
         CHANNELS, HW, REDUCE_SIZE, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_centered, invstd_flat, weight_flat,
         sum_out, vector_out, coeff_mean, coeff_var, coeff_weight,
         CHANNELS, num_tiles, block_tiles, BLOCK_C),
    )

    # For the epilogue, pooled is bf16[BATCH, CHANNELS] (accessed as
    # (n, c) via pooled_offsets = n*C + c). activation is NHWC-flat.
    activation_epilogue = activation_flat_full  # bf16[NUMEL]
    out_flat = torch.empty(NUMEL, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        ((NUMEL + BLOCK_E - 1) // BLOCK_E, 1, 1),
        _epilogue_kernel,
        (pooled_flat, activation_epilogue, mean_flat, invstd_flat,
         weight_flat, bias_flat, scalar_flat,
         coeff_mean, coeff_var, coeff_weight, out_flat,
         CHANNELS, HW, BLOCK_E),
    )
    out = out_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS).permute(0, 3, 1, 2)

    # Return sum, vector, out. mul_10 = sum_centered * weight (arg4).
    # But the reference used mul_10 = sum_2 * arg3.view(CHANNELS), where arg3
    # is invstd, so vector_out = sum_centered * invstd matches mul_10.
    return sum_out, vector_out, out
