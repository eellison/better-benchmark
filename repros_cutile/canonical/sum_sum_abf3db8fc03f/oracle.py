"""cuTile port of sum_sum_abf3db8fc03f: ResNet-style avg-pool -> BN backward.

Matches Triton's 3-kernel structure: split-K partial reduce (produces the
returned `where` tensor and per-tile partial sums), per-channel finalize,
and the BN-backward elementwise epilogue. All reductions and math live
inside cuTile kernels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.0012755102040816326


@ct.kernel
def _partial_reduce_kernel(
    pooled_ptr,      # bf16 [N, C]
    mask_ptr,        # bool [N, C, H, W]
    act_ptr,         # bf16 [N, C, H, W]  any strides
    mean_ptr,        # f32  [C]
    where_out_ptr,   # bf16 [N, C, H, W]  contiguous
    partial_sum_ptr, # f32  [C, NUM_TILES]
    partial_dot_ptr, # f32  [C, NUM_TILES]
    C_C: ct.Constant[int],
    H_C: ct.Constant[int],
    W_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    REDUCE_SIZE: ct.Constant[int],
    NUM_TILES_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    """Grid: (C, num_tiles, 1). One (c, r-tile) partial per program."""
    c = ct.bid(0)
    tile = ct.bid(1)
    r_idx = ct.arange(BLOCK_R, dtype=ct.int32)
    r = tile * BLOCK_R + r_idx
    active = r < REDUCE_SIZE

    n = r // HW_C
    hw = r - n * HW_C
    h = hw // W_C
    w = hw - h * W_C
    c_bc = ct.full((BLOCK_R,), c, dtype=ct.int32)

    pooled = ct.astype(
        ct.gather(pooled_ptr, (n, c_bc), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    divided_bf = ct.astype(pooled / 49.0, ct.bfloat16)
    zero_bf = ct.astype(0.0, ct.bfloat16)
    zero_bc = ct.broadcast_to(ct.reshape(zero_bf, (1,)), (BLOCK_R,))
    pred = ct.gather(mask_ptr, (n, c_bc, h, w), mask=active,
                     padding_value=False)
    where_bf = ct.where(pred, zero_bc, divided_bf)
    ct.scatter(where_out_ptr, (n, c_bc, h, w), where_bf, mask=active)

    where_f = ct.astype(where_bf, ct.float32)
    act = ct.astype(
        ct.gather(act_ptr, (n, c_bc, h, w), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_bc = ct.broadcast_to(mean, (BLOCK_R,))
    centered = act - mean_bc
    product = where_f * centered

    where_masked = ct.where(active, where_f, 0.0)
    product_masked = ct.where(active, product, 0.0)
    partial_sum = ct.sum(where_masked)
    partial_dot = ct.sum(product_masked)
    ct.store(partial_sum_ptr, index=(c, tile),
             tile=ct.reshape(partial_sum, (1, 1)))
    ct.store(partial_dot_ptr, index=(c, tile),
             tile=ct.reshape(partial_dot, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr, # f32 [C, NUM_TILES]
    partial_dot_ptr, # f32 [C, NUM_TILES]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    sum_out_ptr,     # f32 [C]
    vector_out_ptr,  # f32 [C]
    coeff_mean_ptr,  # f32 [C]
    coeff_var_ptr,   # f32 [C]
    coeff_weight_ptr,# f32 [C]
    NUM_TILES_C: ct.Constant[int],
    BLOCK_TILES_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    """Grid: (C // BLOCK_C_,). Reduces partials across tile axis."""
    c_block = ct.bid(0)
    part_sum = ct.astype(
        ct.load(partial_sum_ptr, index=(c_block, 0),
                shape=(BLOCK_C_, BLOCK_TILES_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    part_dot = ct.astype(
        ct.load(partial_dot_ptr, index=(c_block, 0),
                shape=(BLOCK_C_, BLOCK_TILES_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    sum_where = ct.sum(part_sum, axis=1)
    sum_centered = ct.sum(part_dot, axis=1)

    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    scaled_sum = sum_where * SCALE_C
    scaled_centered = sum_centered * SCALE_C
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
    where_ptr,        # bf16 [NUMEL]  contiguous
    act_ptr,          # bf16 [N, C, H, W]  any strides
    mean_ptr,         # f32  [C]
    coeff_mean_ptr,   # f32  [C]
    coeff_var_ptr,    # f32  [C]
    coeff_weight_ptr, # f32  [C]
    out_ptr,          # bf16 [NUMEL]  contiguous
    C_C: ct.Constant[int],
    H_C: ct.Constant[int],
    W_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    NUMEL_C: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    """Grid: (NUMEL // BLOCK_E,)."""
    pid = ct.bid(0)
    where_bf = ct.load(where_ptr, index=(pid,), shape=(BLOCK_E,))
    where_v = ct.astype(where_bf, ct.float32)

    off_idx = ct.arange(BLOCK_E, dtype=ct.int32)
    offsets = pid * BLOCK_E + off_idx
    active = offsets < NUMEL_C
    hw_val = offsets - (offsets // HW_C) * HW_C
    n_ = offsets // (C_C * HW_C)
    c_ = (offsets // HW_C) - n_ * C_C
    h_ = hw_val // W_C
    w_ = hw_val - h_ * W_C

    act = ct.astype(
        ct.gather(act_ptr, (n_, c_, h_, w_), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(
        ct.gather(mean_ptr, c_, mask=active, padding_value=0.0),
        ct.float32,
    )
    coeff_mean = ct.astype(
        ct.gather(coeff_mean_ptr, c_, mask=active, padding_value=0.0),
        ct.float32,
    )
    coeff_var = ct.astype(
        ct.gather(coeff_var_ptr, c_, mask=active, padding_value=0.0),
        ct.float32,
    )
    coeff_weight = ct.astype(
        ct.gather(coeff_weight_ptr, c_, mask=active, padding_value=0.0),
        ct.float32,
    )

    centered = act - mean
    variance_term = centered * coeff_var
    without_var = where_v - variance_term
    without_mean = without_var - coeff_mean
    result = without_mean * coeff_weight
    ct.store(out_ptr, index=(pid,), tile=ct.astype(result, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="78158bd3", BLOCK_R=1024, BLOCK_E=256)
@oracle_impl(hardware="B200", point="0a877abf", BLOCK_R=2048, BLOCK_E=256)
@oracle_impl(hardware="B200", point="186ca521", BLOCK_R=512, BLOCK_E=256)
@oracle_impl(hardware="B200", point="b0156f06", BLOCK_R=1024, BLOCK_E=256)
@oracle_impl(hardware="B200", point="50796f54", BLOCK_R=2048, BLOCK_E=256)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_E):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _s0, _s1 = inputs
    device = arg0_1.device
    n = int(arg2_1.shape[0])
    c = int(arg2_1.shape[1])
    h = int(arg2_1.shape[2])
    w = int(arg2_1.shape[3])
    hw = h * w
    reduce_size = n * hw
    numel = n * c * hw
    num_tiles = (reduce_size + BLOCK_R - 1) // BLOCK_R
    block_tiles = 1 << (num_tiles - 1).bit_length()

    # Materialize scalar bf16 zero output.
    full = torch.zeros((), device=device, dtype=torch.bfloat16)
    where_out = torch.empty_strided(
        (n, c, h, w), _contiguous_stride((n, c, h, w)),
        device=device, dtype=torch.bfloat16,
    )
    out_dense = torch.empty_strided(
        (n, c, h, w), _contiguous_stride((n, c, h, w)),
        device=device, dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((c, num_tiles), device=device, dtype=torch.float32)
    partial_dot = torch.empty((c, num_tiles), device=device, dtype=torch.float32)

    pooled_2d = arg0_1.view(n, c)
    mean_1d = arg3_1.view(c).contiguous()
    invstd_1d = arg4_1.view(c).contiguous()
    weight_1d = arg5_1.view(c).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, num_tiles, 1), _partial_reduce_kernel,
        (pooled_2d, arg1_1, arg2_1, mean_1d, where_out,
         partial_sum, partial_dot,
         c, h, w, hw, reduce_size, num_tiles, BLOCK_R),
    )

    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    vector_out = torch.empty((c,), device=device, dtype=torch.float32)
    coeff_mean = torch.empty((c,), device=device, dtype=torch.float32)
    coeff_var = torch.empty((c,), device=device, dtype=torch.float32)
    coeff_weight = torch.empty((c,), device=device, dtype=torch.float32)

    BLOCK_C = 8
    assert c % BLOCK_C == 0
    ct.launch(
        stream, (c // BLOCK_C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, vector_out, coeff_mean, coeff_var, coeff_weight,
         num_tiles, block_tiles, BLOCK_C, SCALE),
    )

    assert numel % BLOCK_E == 0
    ct.launch(
        stream, (numel // BLOCK_E, 1, 1), _epilogue_kernel,
        (where_out.view(-1), arg2_1, mean_1d,
         coeff_mean, coeff_var, coeff_weight, out_dense.view(-1),
         c, h, w, hw, numel, BLOCK_E),
    )
    return full, where_out, sum_out, vector_out, out_dense
