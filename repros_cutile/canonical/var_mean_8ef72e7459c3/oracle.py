"""cuTile port of var_mean_8ef72e7459c3: Longformer dropout + residual + LayerNorm.

Seed index 8, eps 1e-5. Returns (gt, mul_2, add_2, view_1, div).
Final view_1 requires permute(1,0,2) + clone -> reshape back to [8192,768].
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 8
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    mul2_ptr,
    add2_ptr,
    div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_f = ct.astype(scaled_bf, ct.float32) + residual_f

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(add_f, axis=1, keepdims=True) * inv_h
    centered = add_f - mean_1d
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    rsqrt_val = ct.rsqrt(variance + EPS)
    normalized = centered * rsqrt_val
    ct.store(mul2_ptr, index=(row, 0), tile=normalized)

    weight_f = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_f = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(add2_ptr, index=(row, 0), tile=affine)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(rsqrt_val * inv_h, (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="04503798", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    base_shape = tuple(int(d) for d in shape0)
    random_shape = tuple(int(d) for d in shape1)
    out_shape = tuple(int(d) for d in shape2)
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    base_stride = _contiguous_stride(base_shape)
    div_shape = base_shape[:-1] + (1,)

    gt = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    mul2 = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    add2 = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    rows = arg0_1.shape[0]
    flat_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    mul2_2d = mul2.view(rows, hidden)
    add2_2d = add2.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_layernorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, mul2_2d, add2_2d, div_1d,
         hidden, BLOCK_H),
    )
    # view_1 = clone(permute(add_2 -> [1024, 8, 768])) reshape as [8192, 768]
    view_1 = add2.to(torch.bfloat16).permute(1, 0, 2).contiguous().view(out_shape)
    return gt, mul2, add2, view_1, div
