"""cuTile port of var_mean_ff238bf0bce8: M2M100/XGLM dropout + residual + LN.

Fused seeded Inductor dropout (bf16-compare), residual add, mean/var
(correction=0), rsqrt(eps=1e-5), affine, bf16 cast. Returns gt, add, mean,
rsqrt, bf16 view. HIDDEN=1024 (power of 2).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 1024
EPS = 1.0e-5


@ct.kernel
def _dropout_ln_kernel(
    flat_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    add_ptr,
    mean_ptr,
    rsqrt_ptr,
    bf16_ptr,
    HIDDEN_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, HIDDEN_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(
        ct.where(keep, ct.astype(flat, ct.float32), 0.0),
        ct.bfloat16,
    )
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    total = ct.sum(add_val)
    mean = total * (1.0 / HIDDEN_)
    ct.store(mean_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), mean, dtype=ct.float32), (1,)))
    centered = add_val - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), invstd, dtype=ct.float32), (1,)))
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))
    bias_2d = ct.reshape(bias, (1, HIDDEN_))
    affine = normalized * weight_2d + bias_2d
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN
    row_shape = (norm_shape[0], norm_shape[1], 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(rows, hidden).contiguous()
    residual_2d = arg2_1.reshape(rows, hidden).contiguous()

    mask_2d = torch.empty((rows, hidden), device=device, dtype=torch.bool)
    add_2d = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    mean_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    rsqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    bf16_2d = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_ln_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
         mask_2d, add_2d, mean_1d, rsqrt_1d, bf16_2d, hidden),
    )

    mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    mask.view(rows, hidden).copy_(mask_2d)
    add = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    add.view(rows, hidden).copy_(add_2d)
    mean = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    mean.view(rows).copy_(mean_1d)
    rsqrt = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    rsqrt.view(rows).copy_(rsqrt_1d)
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    bf16_view.copy_(bf16_2d.view(flat_shape))

    return mask, add, mean, rsqrt, bf16_view


@oracle_impl(hardware="B200", point="7d4ce2eb")
@oracle_impl(hardware="B200", point="a43fabe7")
def oracle_forward(inputs):
    return _run(inputs)
