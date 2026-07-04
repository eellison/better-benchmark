"""cuTile port of sum_sum_6b1f51177c85: GhostNet SE-gate BN-backward.

Mirrors Triton's 3-kernel split-K structure:
- _partials_kernel: producer + BN-gated mask + partial sums via ct.sum.
- _finalize_kernel: per-slice-channel reduce partials + compute coefficients.
- _epilogue_kernel: full producer output + BN-backward slice output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 9.964923469387754e-06


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partials_kernel(
    gate_ptr,          # f32 [N*FULL_C] (arg0 flat, ch-last for [N,FULL_C,1,1])
    activation_ptr,    # bf16 [N*FULL_C*HW] flat ch-last
    pool_ptr,          # bf16 [N*FULL_C] flat
    bn_activation_ptr, # bf16 [N*SLICE_C*HW] flat ch-last
    mean_ptr,          # f32 [SLICE_C]
    invstd_ptr,        # f32 [SLICE_C]
    weight_ptr,        # f32 [SLICE_C]
    bias_ptr,          # f32 [SLICE_C]
    scalar_ptr,        # bf16 [1]
    partial_sum_ptr,   # f32 [num_tiles * SLICE_C]
    partial_centered_ptr, # f32 [num_tiles * SLICE_C]
    FULL_C: ct.Constant[int],
    SLICE_C: ct.Constant[int],
    SLICE_START: ct.Constant[int],
    HW: ct.Constant[int],
    REDUCE_SIZE: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)  # in [0, SLICE_C)
    tile = ct.bid(1)
    rows = tile * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < REDUCE_SIZE
    n = rows // HW
    spatial = rows - n * HW
    full_c = c + SLICE_START

    # gate_source: [N, FULL_C, 1, 1] ch-last: storage index = n * FULL_C + full_c
    gate_off = n * FULL_C + full_c
    # activation_full: [N, FULL_C, H, W] ch-last: storage = n * FULL_C * HW + spatial * FULL_C + full_c
    full_off = n * (FULL_C * HW) + spatial * FULL_C + full_c
    # pool_source: [N, FULL_C, 1, 1]: storage = n * FULL_C + full_c
    pool_off = gate_off
    # bn_activation: [N, SLICE_C, H, W] ch-last: storage = n * SLICE_C * HW + spatial * SLICE_C + c
    bn_off = n * (SLICE_C * HW) + spatial * SLICE_C + c

    gate = ct.gather(gate_ptr, gate_off)
    activation_bf = ct.gather(activation_ptr, full_off)
    pool_bf = ct.gather(pool_ptr, pool_off)

    # Producer formula matching Triton
    gate3 = gate + 3.0
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    six_f = ct.full((BLOCK_R,), 6.0, dtype=ct.float32)
    gate_clamped = ct.where(gate3 < zero_f, zero_f, gate3)
    gate_clamped = ct.where(gate_clamped > six_f, six_f, gate_clamped)
    gate_norm = gate_clamped * (1.0 / 6.0)
    gate_bf = ct.astype(gate_norm, ct.bfloat16)
    gated_f = ct.astype(activation_bf, ct.float32) * ct.astype(gate_bf, ct.float32)
    gated_bf = ct.astype(gated_f, ct.bfloat16)
    pooled_f = ct.astype(pool_bf, ct.float32) * (1.0 / 196.0)
    pooled_bf = ct.astype(pooled_f, ct.bfloat16)
    producer_f = ct.astype(gated_bf, ct.float32) + ct.astype(pooled_bf, ct.float32)
    producer_bf = ct.astype(producer_f, ct.bfloat16)

    # BN affine + gate mask
    mean = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    bias = ct.gather(bias_ptr, ct.broadcast_to(c, (1,)))
    mean_s = ct.reshape(mean, ())
    invstd_s = ct.reshape(invstd, ())
    weight_s = ct.reshape(weight, ())
    bias_s = ct.reshape(bias, ())

    bn_val = ct.gather(bn_activation_ptr, bn_off)
    bn_f = ct.astype(bn_val, ct.float32)
    centered = bn_f - mean_s
    affine = centered * invstd_s
    affine = affine * weight_s
    affine = affine + bias_s
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_gate = ct.astype(affine_bf, ct.float32)
    take_producer = affine_gate > zero_f

    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_f = ct.astype(scalar, ct.float32)
    selected = ct.where(take_producer,
                        ct.astype(producer_bf, ct.float32),
                        ct.broadcast_to(scalar_f, (BLOCK_R,)))
    selected = ct.where(active, selected, 0.0)
    centered_active = ct.where(active, centered, 0.0)

    sum_selected = ct.sum(selected)
    sum_centered = ct.sum(selected * centered_active)

    off = tile * SLICE_C + c
    ct.scatter(partial_sum_ptr, ct.broadcast_to(off, (1,)),
               ct.reshape(sum_selected, (1,)))
    ct.scatter(partial_centered_ptr, ct.broadcast_to(off, (1,)),
               ct.reshape(sum_centered, (1,)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,
    partial_centered_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vector_out_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    SLICE_C: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    c = ct.bid(0)
    tiles = ct.arange(BLOCK_TILES, dtype=ct.int32)
    active = tiles < NUM_TILES
    offsets = tiles * SLICE_C + c
    zero_i = ct.zeros((BLOCK_TILES,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)

    ps = ct.gather(partial_sum_ptr, safe_off)
    pc = ct.gather(partial_centered_ptr, safe_off)
    ps = ct.where(active, ps, 0.0)
    pc = ct.where(active, pc, 0.0)
    sum_v = ct.sum(ps)
    sum_c = ct.sum(pc)

    invstd_v = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_v = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_v, ())
    weight = ct.reshape(weight_v, ())
    invstd_sq = invstd * invstd

    scaled_sum = sum_v * REDUCE_SCALE
    scaled_centered = sum_c * REDUCE_SCALE
    coeff_var = scaled_centered * invstd_sq
    coeff_weight = invstd * weight
    vector = sum_c * invstd

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_v, (1,)))
    ct.store(vector_out_ptr, index=(c,), tile=ct.reshape(vector, (1,)))
    ct.store(coeff_mean_ptr, index=(c,), tile=ct.reshape(scaled_sum, (1,)))
    ct.store(coeff_var_ptr, index=(c,), tile=ct.reshape(coeff_var, (1,)))
    ct.store(coeff_weight_ptr, index=(c,), tile=ct.reshape(coeff_weight, (1,)))


@ct.kernel
def _epilogue_kernel(
    gate_ptr,
    activation_ptr,
    pool_ptr,
    bn_activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    coeff_mean_ptr,
    coeff_var_ptr,
    coeff_weight_ptr,
    out_full_ptr,      # bf16 [N*FULL_C*HW] flat ch-last
    out_bn_ptr,        # bf16 [N*SLICE_C*HW] flat ch-last
    FULL_C: ct.Constant[int],
    SLICE_C: ct.Constant[int],
    SLICE_START: ct.Constant[int],
    HW: ct.Constant[int],
    NUMEL: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    linear = pid * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
    active = linear < NUMEL
    zero_i = ct.zeros((BLOCK_E,), dtype=ct.int32)
    safe = ct.where(active, linear, zero_i)

    c = safe - (safe // FULL_C) * FULL_C
    nhw = safe // FULL_C
    spatial = nhw - (nhw // HW) * HW
    n = nhw // HW

    gate_off = n * FULL_C + c
    full_off = n * (FULL_C * HW) + spatial * FULL_C + c
    pool_off = gate_off

    gate = ct.gather(gate_ptr, gate_off)
    activation_bf = ct.gather(activation_ptr, full_off)
    pool_bf = ct.gather(pool_ptr, pool_off)

    zero_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    six_f = ct.full((BLOCK_E,), 6.0, dtype=ct.float32)
    gate3 = gate + 3.0
    gate_clamped = ct.where(gate3 < zero_f, zero_f, gate3)
    gate_clamped = ct.where(gate_clamped > six_f, six_f, gate_clamped)
    gate_norm = gate_clamped * (1.0 / 6.0)
    gate_bf = ct.astype(gate_norm, ct.bfloat16)
    gated_f = ct.astype(activation_bf, ct.float32) * ct.astype(gate_bf, ct.float32)
    gated_bf = ct.astype(gated_f, ct.bfloat16)
    pooled_f = ct.astype(pool_bf, ct.float32) * (1.0 / 196.0)
    pooled_bf = ct.astype(pooled_f, ct.bfloat16)
    producer_f = ct.astype(gated_bf, ct.float32) + ct.astype(pooled_bf, ct.float32)
    producer_bf = ct.astype(producer_f, ct.bfloat16)

    # Write full producer
    ct.scatter(out_full_ptr, linear, producer_bf, mask=active)

    # For slice channels only: compute BN-backward and write out_bn
    in_slice = active & (c >= SLICE_START)
    c_slice = c - SLICE_START
    bn_off = n * (SLICE_C * HW) + spatial * SLICE_C + c_slice
    bn_val = ct.gather(bn_activation_ptr, bn_off)
    bn_f = ct.astype(bn_val, ct.float32)
    mean = ct.gather(mean_ptr, c_slice)
    invstd = ct.gather(invstd_ptr, c_slice)
    weight = ct.gather(weight_ptr, c_slice)
    bias = ct.gather(bias_ptr, c_slice)

    centered = bn_f - mean
    affine = centered * invstd
    affine = affine * weight
    affine = affine + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_gate = ct.astype(affine_bf, ct.float32)
    take_producer = affine_gate > zero_f

    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_f = ct.astype(scalar, ct.float32)
    selected = ct.where(take_producer,
                        ct.astype(producer_bf, ct.float32),
                        ct.broadcast_to(scalar_f, (BLOCK_E,)))
    coeff_mean = ct.gather(coeff_mean_ptr, c_slice)
    coeff_var = ct.gather(coeff_var_ptr, c_slice)
    coeff_weight = ct.gather(coeff_weight_ptr, c_slice)

    variance_term = centered * coeff_var
    without_var = selected - variance_term
    without_mean = without_var - coeff_mean
    result = without_mean * coeff_weight
    result_bf = ct.astype(result, ct.bfloat16)

    # Compact bn output storage: (n, c_slice, h, w) -> flat linear index
    bn_out_off = bn_off  # same layout as bn_activation (ch-last)
    ct.scatter(out_bn_ptr, bn_out_off, result_bf, mask=in_slice)


@oracle_impl(hardware="B200", point="930e2c0b", BLOCK_R=256, BLOCK_E=256)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_E: int):
    (
        gate_source, activation_full, pool_source, bn_activation,
        mean, invstd, weight, bias, scalar, _shape0,
    ) = inputs

    n, full_c, h, w = (int(d) for d in activation_full.shape)
    slice_c = int(bn_activation.shape[1])
    slice_start = full_c - slice_c
    hw = h * w
    reduce_size = n * hw
    numel = n * full_c * hw
    num_tiles = (reduce_size + BLOCK_R - 1) // BLOCK_R
    block_tiles = _next_power_of_2(num_tiles)

    device = activation_full.device
    partial_sum = torch.empty((num_tiles * slice_c,), device=device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles * slice_c,), device=device, dtype=torch.float32)
    sum_out = torch.empty((slice_c,), device=device, dtype=torch.float32)
    vector_out = torch.empty((slice_c,), device=device, dtype=torch.float32)
    coeff_mean = torch.empty((slice_c,), device=device, dtype=torch.float32)
    coeff_var = torch.empty((slice_c,), device=device, dtype=torch.float32)
    coeff_weight = torch.empty((slice_c,), device=device, dtype=torch.float32)
    out_full = torch.empty_strided(
        tuple(activation_full.shape),
        tuple(int(s) for s in activation_full.stride()),
        device=device, dtype=torch.bfloat16,
    )
    out_bn = torch.empty_strided(
        tuple(bn_activation.shape),
        tuple(int(s) for s in bn_activation.stride()),
        device=device, dtype=torch.bfloat16,
    )

    def _flat_cl(t):
        return torch.as_strided(t, (t.numel(),), (1,), 0)

    # gate_source is [N, FULL_C, 1, 1] f32 contig — storage n * FULL_C + c
    gate_flat = gate_source.contiguous().view(-1)
    pool_flat = pool_source.contiguous().view(-1)
    act_flat = _flat_cl(activation_full)
    bn_flat = _flat_cl(bn_activation)
    outfull_flat = _flat_cl(out_full)
    outbn_flat = _flat_cl(out_bn)

    mean_1d = mean.view(-1)
    invstd_1d = invstd.view(-1)
    weight_1d = weight.view(-1)
    bias_1d = bias.view(-1)
    scalar_1d = scalar.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (slice_c, num_tiles, 1),
        _partials_kernel,
        (gate_flat, act_flat, pool_flat, bn_flat,
         mean_1d, invstd_1d, weight_1d, bias_1d, scalar_1d,
         partial_sum, partial_centered,
         full_c, slice_c, slice_start, hw, reduce_size, BLOCK_R),
    )
    ct.launch(
        stream, (slice_c, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_centered, invstd_1d, weight_1d,
         sum_out, vector_out, coeff_mean, coeff_var, coeff_weight,
         slice_c, num_tiles, block_tiles),
    )
    ct.launch(
        stream, ((numel + BLOCK_E - 1) // BLOCK_E, 1, 1),
        _epilogue_kernel,
        (gate_flat, act_flat, pool_flat, bn_flat,
         mean_1d, invstd_1d, weight_1d, bias_1d, scalar_1d,
         coeff_mean, coeff_var, coeff_weight,
         outfull_flat, outbn_flat,
         full_c, slice_c, slice_start, hw, numel, BLOCK_E),
    )
    return out_full, sum_out, vector_out, out_bn
