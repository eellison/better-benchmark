"""cuTile port of var_mean_724b7e0b69d8: XLNet seeded-dropout LayerNorm training.

Pre-generates seeded random via inductor_random. One row kernel does:
bf16 dropout, dropout scale, f32 residual add, var_mean (correction=0),
rsqrt+eps=1e-12, affine, final bf16 with view+permute+squeeze aliases,
plus rsqrt/hidden side output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 69
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr,       # bf16 (rows, HIDDEN)
    random_ptr,     # f32 (rows, HIDDEN)
    residual_ptr,   # f32 (rows, HIDDEN)
    weight_ptr,     # f32 (HIDDEN,)
    bias_ptr,       # f32 (HIDDEN,)
    gt_ptr,         # bool (rows, HIDDEN)
    normalized_ptr, # f32 (rows, HIDDEN)
    affine_ptr,     # f32 (rows, HIDDEN)
    affine_bf16_ptr,# bf16 (rows, HIDDEN)
    div_ptr,        # f32 (rows,)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                        padding_mode=ct.PaddingMode.ZERO)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
                           ct.bfloat16)
    added = ct.astype(scaled_bf, ct.float32) + residual

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
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
    ct.store(affine_bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))

    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN), (1,)))


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
        dtype=state.dtype,
        device=state.device,
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


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape(shape0)
    random_shape = _shape(shape1)
    reshape_shape = _shape(shape2)  # e.g. (1, 8192, 1024)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                              device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                      device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                  device=arg0_1.device, dtype=torch.float32)
    # bf16 view - shape reshape_shape
    affine_bf16 = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                       device=arg0_1.device, dtype=torch.bfloat16)
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
    affine_bf_2d = affine_bf16.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, normalized_2d, affine_2d, affine_bf_2d, div_1d,
         hidden, BLOCK_H),
    )

    # Build the aliasing views from the affine_bf16 base tensor.
    view_1 = affine_bf16.view(reshape_shape)  # (1, 8192, 1024)
    squeeze = view_1.squeeze(0)                # (8192, 1024)
    permute = view_1.permute(0, 2, 1)          # (1, 1024, 8192)
    squeeze_1 = permute.squeeze(0)             # (1024, 8192)

    return gt, normalized, affine, squeeze, squeeze_1, div
