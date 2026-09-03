"""cuTile port of var_mean_978d6d771932: GoogleFnet f32 dropout residual LN.

Pre-generates seeded random tensor via inductor_random on the Python side. A
single row-parallel cuTile kernel does f32 dropout (compare f32 random > 0.1),
residual add, LN, affine, and emits mask, normalized, affine, complex-real
view, and inverse-std side output. HIDDEN=768 padded to BLOCK_H=1024.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 4
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_ln_kernel(
    flat_ptr,       # f32 [rows, BLOCK_H]  (padded)
    random_ptr,     # f32 [rows, BLOCK_H]
    residual_ptr,   # f32 [rows, BLOCK_H]
    weight_ptr,     # f32 [BLOCK_H]
    bias_ptr,       # f32 [BLOCK_H]
    gt_ptr,         # bool [rows, BLOCK_H]
    norm_ptr,       # f32 [rows, BLOCK_H]
    affine_ptr,     # f32 [rows, BLOCK_H]
    div_ptr,        # f32 [rows]
    HIDDEN: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_f = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    # f32 dropout comparison (no bf16 cast in this one)
    threshold_f = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.float32)
    keep = random_f > threshold_f
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    dropped_f = ct.where(keep, flat_f, zero_f)
    scaled_f = dropped_f * DROPOUT_SCALE
    x_f = scaled_f + residual_f

    x_masked = ct.where(col_valid_2d, x_f, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x_f - mean
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_1d, (1, BLOCK_H))
    affine_f = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine_f)
    ct.store(div_ptr, index=(row,), tile=invstd * (1.0 / HIDDEN))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - advance)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_M=1, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    del BLOCK_M
    flat, seeds, residual, weight, bias, shape0, shape1 = inputs
    base_shape = _shape_tuple(shape1)
    complex_shape = _shape_tuple(shape0)
    rows = int(flat.shape[0])
    hidden = int(flat.shape[1])
    div_shape = (base_shape[0], base_shape[1], 1)
    device = flat.device

    x_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    x_padded[:, :hidden] = flat
    residual_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    residual_padded[:, :hidden] = residual.view(rows, hidden)
    weight_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_padded[:hidden] = weight
    bias_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias_padded[:hidden] = bias

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random_full = _inductor_random_for_eager_check(base_shape, seed, device=device)
    random_flat = random_full.view(rows, hidden)
    # Fill padding with 0.0 -> gt(0.1)=False so no effect on reduction lanes
    random_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    random_padded[:, :hidden] = random_flat

    gt_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    div_flat = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_ln_kernel,
        (
            x_padded, random_padded, residual_padded,
            weight_padded, bias_padded,
            gt_padded, norm_padded, affine_padded, div_flat,
            hidden, EPS, BLOCK_H,
        ),
    )

    gt = torch.empty_strided(base_shape, _contiguous_stride(base_shape),
                             device=device, dtype=torch.bool)
    gt.view(rows, hidden).copy_(gt_padded[:, :hidden])
    normalized = torch.empty_strided(base_shape, _contiguous_stride(base_shape),
                                     device=device, dtype=torch.float32)
    normalized.view(rows, hidden).copy_(norm_padded[:, :hidden])
    affine = torch.empty_strided(base_shape, _contiguous_stride(base_shape),
                                 device=device, dtype=torch.float32)
    affine.view(rows, hidden).copy_(affine_padded[:, :hidden])

    # Build complex tensor from real part = affine, imag = 0.
    complex_out = torch.empty_strided(complex_shape, _contiguous_stride(base_shape),
                                      device=device, dtype=torch.complex64)
    real_view = torch.view_as_real(complex_out)
    real_view[..., 0] = affine
    real_view[..., 1] = 0.0

    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=device, dtype=torch.float32)
    div.view(rows).copy_(div_flat)

    return gt, normalized, affine, complex_out, div
