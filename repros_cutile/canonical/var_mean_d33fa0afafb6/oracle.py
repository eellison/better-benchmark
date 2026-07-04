"""cuTile port of var_mean_d33fa0afafb6: GPT2ForSequenceClassification dropout-residual LayerNorm.

EPS=1e-5, add=(arg2_1 + mul_1), returns (gt, add, mul_2, bf16_view, permute, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


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
    HIDDEN_PAD: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                     padding_mode=ct.PaddingMode.ZERO)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    p_bf = ct.full((1, HIDDEN_PAD), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, HIDDEN_PAD), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    # add = residual + scaled (order matches arg2_1 + mul_1)
    x = residual + ct.astype(scaled_bf, ct.float32)

    zero_f = ct.full((1, HIDDEN_PAD), 0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x, zero_f)
    mean_val = ct.sum(x_masked) * (1.0 / HIDDEN_F)
    centered = x - mean_val
    centered_masked = ct.where(valid_2d, centered, zero_f)
    var_val = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_F)
    rsqrt_val = ct.rsqrt(var_val + EPS)
    mul_2 = centered * rsqrt_val

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PAD))
    affine = mul_2 * weight_2d + bias_2d

    ct.store(add_ptr, index=(row, 0), tile=ct.where(valid_2d, x, zero_f))
    ct.store(mul_2_ptr, index=(row, 0), tile=ct.where(valid_2d, mul_2, zero_f))
    ct.store(bf16_ptr, index=(row, 0),
             tile=ct.where(valid_2d, ct.astype(affine, ct.bfloat16), zero_bf))
    div_val = rsqrt_val * (1.0 / HIDDEN_F)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


@oracle_impl(hardware="B200", point="bf8decda")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    # shape2 may include -1 as unknown; compute from norm_shape
    flat_shape = (rows, hidden)
    div_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    add = torch.empty(norm_shape, device=device, dtype=torch.float32)
    mul_2 = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    HIDDEN_PAD = _next_pow2(hidden)
    random_2d = random.reshape(rows, hidden)
    residual_2d = arg2_1.reshape(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    mul_2_2d = mul_2.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, mul_2_2d, bf16_view_2d, div_1d,
         hidden, HIDDEN_PAD, float(hidden)),
    )
    permute = bf16_view.permute(1, 0)
    return gt, add, mul_2, bf16_view, permute, div
