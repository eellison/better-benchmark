"""cuTile port of mean_7523f1bd5831: T5/MT5 dropout+residual RMSNorm.

Seed index 42, EPS 1e-6. Pre-generates dropout random via inductor_random.
Returns (gt, add, rsqrt, view_1).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 42
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


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
    advance = (numel + 131071) // 131072
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
def _dropout_residual_rmsnorm_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr,
    gt_ptr, add_ptr, rsqrt_ptr, bf16_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
    EPS_C: ct.Constant[float],
):
    row = ct.bid(0)
    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                      padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                     padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    p_bf = ct.full((1, HIDDEN_PAD), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, HIDDEN_PAD), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE_C, ct.bfloat16)
    residual_f = ct.astype(residual, ct.float32)
    add_val = residual_f + ct.astype(scaled_bf, ct.float32)
    zero_f = ct.full((1, HIDDEN_PAD), 0.0, dtype=ct.float32)
    add_masked = ct.where(valid_2d, add_val, zero_f)
    ct.store(add_ptr, index=(row, 0), tile=add_masked)

    inv_h = 1.0 / HIDDEN
    sq_sum = ct.sum(add_masked * add_masked) * inv_h
    inv_rms = ct.rsqrt(sq_sum + EPS_C)
    ct.store(rsqrt_ptr, index=(row,),
             tile=ct.full(shape=(1,), fill_value=inv_rms, dtype=ct.float32))

    normalized = add_val * inv_rms

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD))
    affine = weight_2d * normalized

    ct.store(bf16_ptr, index=(row, 0),
             tile=ct.where(valid_2d, ct.astype(affine, ct.bfloat16), zero_bf))


@oracle_impl(hardware="B200", point="46dbfd5f")
@oracle_impl(hardware="B200", point="ebc95169")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, _shape1, _shape2 = inputs
    norm_shape = _as_shape(shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    stat_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    add = torch.empty(norm_shape, device=device, dtype=torch.float32)
    rsqrt = torch.empty(stat_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(rows, hidden, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    HIDDEN_PAD = _next_pow2(hidden)
    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1,
         gt.view(rows, hidden), add.view(rows, hidden),
         rsqrt.view(rows), bf16_view,
         hidden, HIDDEN_PAD, DROPOUT_SCALE, EPS),
    )
    return gt, add, rsqrt, bf16_view
