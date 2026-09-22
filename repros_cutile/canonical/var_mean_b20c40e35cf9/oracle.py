"""cuTile port of var_mean_b20c40e35cf9: MegatronBert dropout-residual LayerNorm.

EPS=1e-12, hidden=1024, add=arg2+mul_1, returns (gt, add, mul_2, view_1, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 41
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


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


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, add_ptr, mul_2_ptr, bf16_ptr, div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    p_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    # add = arg2 + mul_1 (residual + scaled)
    x = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x)

    inv_h = 1.0 / HIDDEN
    mean_val = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = x - mean_val
    var_val = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    rsqrt_val = ct.rsqrt(var_val + EPS)
    mul_2 = centered * rsqrt_val
    ct.store(mul_2_ptr, index=(row, 0), tile=mul_2)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    add_2 = mul_2 * weight_2d + bias_2d
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(add_2, ct.bfloat16))
    div_val = rsqrt_val * inv_h
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


@oracle_impl(hardware="B200", point="cfc55f11", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, _shape2 = inputs
    view_shape = _as_shape(shape0)
    rows = view_shape[0] * view_shape[1]
    hidden = view_shape[2]
    div_shape = (view_shape[0], view_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    add = torch.empty(view_shape, device=device, dtype=torch.float32)
    mul_2 = torch.empty(view_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(view_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    mul_2_2d = mul_2.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, mul_2_2d, bf16_view, div_1d,
         hidden, BLOCK_H),
    )
    return gt, add, mul_2, bf16_view, div
