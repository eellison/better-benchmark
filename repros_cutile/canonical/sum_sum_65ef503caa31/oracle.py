"""cuTile port of sum_sum_65ef503caa31: phlippe DenseNet BN-backward split-K.

Mirrors Triton's 3-kernel structure:
- _partial_reduce_kernel: per-channel partial sums via `ct.sum(..., axis=0)`.
- _finalize_kernel: reduce partials, compute finalized coefficients.
- _epilogue_kernel: elementwise BN-backward + slice-add epilogue (bf16).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 3.0517578125e-05
SLICE_START = 16


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,       # bf16 [N*C*HW] flat contig
    fill_ptr,       # bf16 [1]
    rhs_ptr,        # bf16 [N*C*HW] flat contig
    activation_ptr, # bf16 [N*C*HW] flat contig
    mean_ptr,       # f32 [C]
    partial_sum_ptr,   # f32 [num_k_tiles * C]
    partial_dot_ptr,   # f32 [num_k_tiles * C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    k = tile * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL
    n = k // HW
    hw = k - n * HW
    # Compact NCHW: n * C * HW + c * HW + hw
    offsets = n * (C * HW) + c * HW + hw

    gate_bf = ct.gather(mask_ptr, offsets)
    rhs_bf = ct.gather(rhs_ptr, offsets)
    gate = ct.astype(gate_bf, ct.float32)
    rhs = ct.astype(rhs_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    zero_f = ct.zeros((BLOCK_K,), dtype=ct.float32)
    selected = ct.where(gate <= zero_f,
                        ct.broadcast_to(fill_f, (BLOCK_K,)), rhs)

    activation_bf = ct.gather(activation_ptr, offsets)
    activation = ct.astype(activation_bf, ct.float32)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean_bc = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    centered = activation - mean_bc

    selected_active = ct.where(active, selected, 0.0)
    centered_active = ct.where(active, centered, 0.0)
    sum_v = ct.sum(selected_active)
    dot_v = ct.sum(selected_active * centered_active)

    partial_off = tile * C + c
    ct.scatter(partial_sum_ptr, ct.broadcast_to(partial_off, (1,)),
               ct.reshape(sum_v, (1,)))
    ct.scatter(partial_dot_ptr, ct.broadcast_to(partial_off, (1,)),
               ct.reshape(dot_v, (1,)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    C: ct.Constant[int],
    NUM_K_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    tile_offsets = ct.arange(BLOCK_TILES, dtype=ct.int32)
    active = tile_offsets < NUM_K_TILES
    offsets = tile_offsets * C + c
    zero_i = ct.zeros((BLOCK_TILES,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)

    sum_vals = ct.gather(partial_sum_ptr, safe_off)
    dot_vals = ct.gather(partial_dot_ptr, safe_off)
    sum_vals = ct.where(active, sum_vals, 0.0)
    dot_vals = ct.where(active, dot_vals, 0.0)
    sum_v = ct.sum(sum_vals)
    dot_v = ct.sum(dot_vals)

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, ())
    weight = ct.reshape(weight_scalar, ())
    invstd_sq = invstd * invstd
    dot_mean = dot_v * SCALE
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_v, (1,)))
    ct.store(scaled_dot_out_ptr, index=(c,), tile=ct.reshape(dot_v * invstd, (1,)))
    ct.store(mean_term_ptr, index=(c,), tile=ct.reshape(sum_v * SCALE, (1,)))
    ct.store(correction_scale_ptr, index=(c,), tile=ct.reshape(correction_scale, (1,)))
    ct.store(output_scale_ptr, index=(c,), tile=ct.reshape(output_scale, (1,)))


@ct.kernel
def _epilogue_kernel(
    sliced_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    mean_term_ptr,
    correction_scale_ptr,
    output_scale_ptr,
    out_ptr,
    C: ct.Constant[int],
    INPUT_C: ct.Constant[int],
    HW: ct.Constant[int],
    NUMEL: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
    active = offsets < NUMEL
    zero_i = ct.zeros((BLOCK_E,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)

    hw_idx = safe_off - (safe_off // HW) * HW
    tmp = safe_off // HW
    channel = tmp - (tmp // C) * C
    n = tmp // C
    slice_off = n * (INPUT_C * HW) + (channel + SLICE_START) * HW + hw_idx

    gate_bf = ct.gather(mask_ptr, safe_off)
    rhs_bf = ct.gather(rhs_ptr, safe_off)
    gate = ct.astype(gate_bf, ct.float32)
    rhs = ct.astype(rhs_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    zero_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    selected = ct.where(gate <= zero_f,
                        ct.broadcast_to(fill_f, (BLOCK_E,)), rhs)

    activation_bf = ct.gather(activation_ptr, safe_off)
    activation = ct.astype(activation_bf, ct.float32)
    mean = ct.gather(mean_ptr, channel)
    centered = activation - mean
    correction_scale = ct.gather(correction_scale_ptr, channel)
    correction = centered * correction_scale
    after_correction = selected - correction
    mean_term = ct.gather(mean_term_ptr, channel)
    centered_grad = after_correction - mean_term
    output_scale = ct.gather(output_scale_ptr, channel)
    grad = centered_grad * output_scale
    grad_bf = ct.astype(grad, ct.bfloat16)

    sliced_bf = ct.gather(sliced_ptr, slice_off)
    add_f = ct.astype(sliced_bf, ct.float32) + ct.astype(grad_bf, ct.float32)
    add_bf = ct.astype(add_f, ct.bfloat16)
    ct.scatter(out_ptr, offsets, add_bf, mask=active)


@oracle_impl(hardware="B200", point="4363851f", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="71c373f0", BLOCK_K=1024)
def oracle_forward(inputs, *, BLOCK_K: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    device = arg1.device
    n, c, h, w = arg1.shape
    hw = h * w
    input_c = int(arg0.shape[1])
    numel = n * c * hw
    k_total = n * hw
    num_k_tiles = (k_total + BLOCK_K - 1) // BLOCK_K
    block_tiles = _next_power_of_2(num_k_tiles)
    BLOCK_E = 512

    partial_sum = torch.empty((num_k_tiles * c,), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles * c,), device=device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((c,), device=device, dtype=torch.float32)
    mean_term = torch.empty((c,), device=device, dtype=torch.float32)
    correction_scale = torch.empty((c,), device=device, dtype=torch.float32)
    output_scale = torch.empty((c,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (n, c, h, w), (c * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )

    def _flat(t):
        size = t.untyped_storage().size() // 2
        return torch.as_strided(t, (size,), (1,), 0)

    arg0_flat = _flat(arg0)
    arg1_flat = _flat(arg1)
    arg3_flat = _flat(arg3)
    arg4_flat = _flat(arg4)
    add_flat = _flat(add_out)
    fill_1d = arg2.view(1)
    mean_1d = arg5.view(c)
    invstd_1d = arg6.view(c)
    weight_1d = arg7.view(c)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, num_k_tiles, 1),
        _partial_reduce_kernel,
        (arg1_flat, arg2.view(1), arg3_flat, arg4_flat, mean_1d,
         partial_sum, partial_dot, c, hw, k_total, BLOCK_K),
    )
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scaled_dot, mean_term, correction_scale, output_scale,
         c, num_k_tiles, block_tiles),
    )
    ct.launch(
        stream,
        ((numel + BLOCK_E - 1) // BLOCK_E, 1, 1),
        _epilogue_kernel,
        (arg0_flat, arg1_flat, fill_1d, arg3_flat, arg4_flat, mean_1d,
         mean_term, correction_scale, output_scale, add_flat,
         c, input_c, hw, numel, BLOCK_E),
    )
    return sum_out, scaled_dot, add_out, add_out[:, :SLICE_START, :, :]
