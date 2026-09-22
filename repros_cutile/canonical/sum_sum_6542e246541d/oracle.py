"""cuTile port of sum_sum_6542e246541d: GhostNet BN-backward + channel slice.

Mirrors Triton's 3-kernel split-K:
- _partial_reduce_kernel: per-channel partial sums via `ct.sum(..., axis=0)`.
- _finalize_kernel: reduce partials + compute coefficients.
- _epilogue_kernel: full elementwise BN-backward output (channels-last bf16).

Inputs are channels-last strided. C=24, IN_C=48, SLICE_OFFSET=24.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 24
IN_C = 48
SLICE_OFFSET = 24
H = 112
W = 112
HW = H * W
K_TOTAL = N * HW
TOTAL = K_TOTAL * C
SCALE = 1.5570192920918366e-07


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    arg0_ptr,       # bf16 [N*IN_C*HW] flat (channels-last)
    arg1_ptr,       # bf16 [N*C*HW] flat (channels-last)
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    fill_ptr,       # bf16 [1]
    partial_sum_ptr,   # f32 [num_k_blocks * C]
    partial_dot_ptr,   # f32 [num_k_blocks * C]
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    k_block = ct.bid(1)
    k = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL
    # channels-last (spatial-major): storage index = k * C + c for arg1
    x_offsets = k * C + c
    upstream_offsets = k * IN_C + (SLICE_OFFSET + c)

    upstream_bf = ct.gather(arg0_ptr, upstream_offsets)
    x_bf = ct.gather(arg1_ptr, x_offsets)
    mean_v = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    invstd_v = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_v = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    bias_v = ct.gather(bias_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.reshape(mean_v, ())
    invstd = ct.reshape(invstd_v, ())
    weight = ct.reshape(weight_v, ())
    bias = ct.reshape(bias_v, ())

    x = ct.astype(x_bf, ct.float32)
    centered = x - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine = scaled + bias
    rounded_affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    upstream_f = ct.astype(upstream_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_K,), dtype=ct.float32)
    selected = ct.where(rounded_affine <= zero_f,
                        ct.broadcast_to(fill_f, (BLOCK_K,)),
                        upstream_f)
    selected = ct.where(active, selected, 0.0)
    centered_active = ct.where(active, centered, 0.0)

    sum_v = ct.sum(selected)
    dot_v = ct.sum(selected * centered_active)

    partial_off = k_block * C + c
    ct.scatter(partial_sum_ptr, ct.broadcast_to(partial_off, (1,)),
               ct.reshape(sum_v, (1,)))
    ct.scatter(partial_dot_ptr, ct.broadcast_to(partial_off, (1,)),
               ct.reshape(dot_v, (1,)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scaled_dot_ptr,
    NUM_K_BLOCKS: ct.Constant[int],
    BLOCK_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    block_offsets = ct.arange(BLOCK_BLOCKS, dtype=ct.int32)
    active = block_offsets < NUM_K_BLOCKS
    offsets = block_offsets * C + c
    zero_i = ct.zeros((BLOCK_BLOCKS,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)

    sum_vals = ct.gather(partial_sum_ptr, safe_off)
    dot_vals = ct.gather(partial_dot_ptr, safe_off)
    sum_vals = ct.where(active, sum_vals, 0.0)
    dot_vals = ct.where(active, dot_vals, 0.0)
    sum_v = ct.sum(sum_vals)
    dot_v = ct.sum(dot_vals)

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, ())

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_v, (1,)))
    ct.store(dot_tmp_ptr, index=(c,), tile=ct.reshape(dot_v, (1,)))
    ct.store(scaled_dot_ptr, index=(c,), tile=ct.reshape(dot_v * invstd, (1,)))


@ct.kernel
def _epilogue_kernel(
    arg0_ptr,       # bf16 [N*IN_C*HW]
    arg1_ptr,       # bf16 [N*C*HW]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    fill_ptr,       # bf16 [1]
    sum_ptr,        # f32 [C]
    dot_ptr,        # f32 [C]
    out_ptr,        # bf16 [N*C*HW]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < TOTAL
    c = offsets - (offsets // C) * C
    k = offsets // C

    x_offsets = offsets
    upstream_offsets = k * IN_C + (SLICE_OFFSET + c)

    upstream_bf = ct.gather(arg0_ptr, upstream_offsets)
    x_bf = ct.gather(arg1_ptr, x_offsets)
    mean = ct.gather(mean_ptr, c)
    invstd = ct.gather(invstd_ptr, c)
    weight = ct.gather(weight_ptr, c)
    bias = ct.gather(bias_ptr, c)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    sum_v = ct.gather(sum_ptr, c)
    dot_v = ct.gather(dot_ptr, c)

    x = ct.astype(x_bf, ct.float32)
    centered = x - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine = scaled + bias
    rounded_affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    upstream_f = ct.astype(upstream_bf, ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    selected = ct.where(rounded_affine <= zero_f,
                        ct.broadcast_to(fill_f, (BLOCK,)), upstream_f)

    mean_term = sum_v * SCALE
    dot_scaled = dot_v * SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight
    after_variance = selected - centered * variance_term
    after_mean = after_variance - mean_term
    out_f = after_mean * output_scale
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.scatter(out_ptr, offsets, out_bf, mask=active)


@oracle_impl(hardware="B200", point="4ecbdfd8", BLOCK_K=1024, EPILOGUE_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_K: int, EPILOGUE_BLOCK: int):
    arg0, arg1, arg2, arg3, arg4, arg5, fill = inputs
    device = arg0.device
    num_k_blocks = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    block_blocks = _next_power_of_2(num_k_blocks)

    partial_sum = torch.empty((num_k_blocks * C,), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_blocks * C,), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((C,), device=device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(arg1.shape), tuple(arg1.stride()),
        device=device, dtype=torch.bfloat16,
    )

    def _flat_cl(t):
        return torch.as_strided(t, (t.numel(),), (1,), 0)

    arg0_flat = _flat_cl(arg0)
    arg1_flat = _flat_cl(arg1)
    out_flat = _flat_cl(out)

    mean_1d = arg2.view(C)
    invstd_1d = arg3.view(C)
    weight_1d = arg4.view(C)
    bias_1d = arg5.view(C)
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, num_k_blocks, 1),
        _partial_reduce_kernel,
        (arg0_flat, arg1_flat, mean_1d, invstd_1d, weight_1d, bias_1d, fill_1d,
         partial_sum, partial_dot, BLOCK_K),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, sum_out, dot_tmp, scaled_dot,
         num_k_blocks, block_blocks),
    )
    ct.launch(
        stream,
        ((TOTAL + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK, 1, 1),
        _epilogue_kernel,
        (arg0_flat, arg1_flat, mean_1d, invstd_1d, weight_1d, bias_1d,
         fill_1d, sum_out, dot_tmp, out_flat, EPILOGUE_BLOCK),
    )
    return sum_out, scaled_dot, out
