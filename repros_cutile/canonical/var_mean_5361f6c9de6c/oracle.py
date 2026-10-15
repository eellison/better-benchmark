"""cuTile port of var_mean_5361f6c9de6c: GoogleFnet f32 dropout+residual LayerNorm (seed 5)."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, normalized_ptr, affine_ptr, div_ptr,
    HIDDEN: ct.Constant[int], BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                    padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                        padding_mode=ct.PaddingMode.ZERO)
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    # f32 dropout — comparison in f32.
    keep = rand_f > DROPOUT_P
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    dropped = ct.where(keep, flat, zero_f)
    scaled = dropped * DROPOUT_SCALE
    added = scaled + residual

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    added_masked = ct.where(col_valid, added, zero_f)

    mean = ct.sum(added_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = added - mean
    centered_masked = ct.where(col_valid, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd
    ct.store(normalized_ptr, index=(row, 0), tile=normalized)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                      padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                    padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(div_ptr, index=(row,),
             tile=ct.reshape(invstd * (1.0 / HIDDEN), (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape(shape):
    return tuple(int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype, device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
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
        * curand4_engine_calls * 2
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


@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape(shape0)
    random_shape = _shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                              device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                      device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                  device=arg0_1.device, dtype=torch.float32)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                               device=arg0_1.device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed,
                                              device=arg0_1.device)

    flat_2d = arg0_1.contiguous().view(rows, hidden)
    residual_2d = arg2_1.view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, normalized_2d, affine_2d, div_1d,
         hidden, BLOCK_H),
    )

    # Complex64 view: real=affine, imag=0 with same layout.
    complex_out = affine.to(torch.complex64)

    return gt, normalized, affine, complex_out, div
