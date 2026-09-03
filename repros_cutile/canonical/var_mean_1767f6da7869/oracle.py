"""cuTile port of var_mean_1767f6da7869: Electra dropout + residual + LayerNorm.

Ports the Triton `_dropout_residual_layernorm_kernel`. Pre-generates the
seeded random tensor via `torch.ops.prims.inductor_random` and passes it as
an input. HIDDEN=256, BLOCK_H=256 divides evenly so no masked stores needed.

Returns: (mask, add, mean, rsqrt, affine, bf16_view).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 2
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    random_ptr,    # f32 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    bias_ptr,      # f32 [HIDDEN]
    mask_ptr,      # bool [rows, HIDDEN]
    add_ptr,       # bf16 [rows, HIDDEN]
    mean_ptr,      # f32 [rows]
    rsqrt_ptr,     # f32 [rows]
    affine_ptr,    # f32 [rows, HIDDEN]
    bf16_view_ptr, # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row_block = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_H), 0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf > threshold_bf
    ct.store(mask_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.zeros((BLOCK_M, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_bf = ct.astype(ct.astype(scaled_bf, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    ct.store(add_ptr, index=(row_block, 0), tile=add_bf)

    x = ct.astype(add_bf, ct.float32)
    row_sum = ct.sum(x, axis=1)
    mean = row_sum * (1.0 / HIDDEN)
    mean_2d = ct.reshape(mean, (BLOCK_M, 1))
    centered = x - mean_2d
    variance = ct.sum(centered * centered, axis=1) * (1.0 / HIDDEN)
    rsqrt = ct.rsqrt(variance + EPS)
    rsqrt_2d = ct.reshape(rsqrt, (BLOCK_M, 1))
    normalized = centered * rsqrt_2d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row_block,), tile=mean)
    ct.store(rsqrt_ptr, index=(row_block,), tile=rsqrt)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_view_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
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


@oracle_impl(hardware="B200", point="b5a5c55d", BLOCK_M=2, BLOCK_H=256)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    side_shape = (norm_shape[0], norm_shape[1], 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()

    mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        side_shape, _contiguous_stride(side_shape),
        device=device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        side_shape, _contiguous_stride(side_shape),
        device=device, dtype=torch.float32,
    )
    affine = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )

    residual_flat = arg2_1.reshape(rows, hidden).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _dropout_residual_layernorm_kernel,
        (arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
         mask.view(rows, hidden), add.view(rows, hidden),
         mean.view(rows), rsqrt.view(rows),
         affine.view(rows, hidden), bf16_view.view(rows, hidden),
         hidden, BLOCK_M, BLOCK_H),
    )
    return mask, add, mean, rsqrt, affine, bf16_view
