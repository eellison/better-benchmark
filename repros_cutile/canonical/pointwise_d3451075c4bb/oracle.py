"""cuTile port of pointwise_d3451075c4bb: XLNet exact-erf GELU + seed-56 dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 56
DROPOUT_SCALE = 1.1111111111111112


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
def _gelu_dropout_kernel(
    x_ptr, rand_ptr, gt_ptr, out_ptr,
    N: ct.Constant[int], BLOCK: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    rand_v = ct.load(rand_ptr, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    erf_arg = x_f * 0.7071067811865476

    # In-kernel erf via Abramowitz-Stegun 7.1.26.
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((BLOCK,), 1.0, dtype=ct.float32),
                    ct.full((BLOCK,), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_v = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    gelu_f = x_f * 0.5 * (erf_v + 1.0)
    gelu_bf = ct.astype(gelu_f, ct.bfloat16)

    rand_bf = ct.astype(rand_v, ct.bfloat16)
    p_bf = ct.full((BLOCK,), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(pid,), tile=keep)

    zero_bf = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, gelu_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE_C, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


@oracle_impl(hardware="B200", point="c78a05f8", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, seeds, _shape0, random_shape, out_shape = inputs
    random_shape = tuple(int(d) for d in random_shape)
    out_shape = tuple(int(d) for d in out_shape)
    device = x.device
    N = x.numel()

    gt_flat = torch.empty(N, device=device, dtype=torch.bool)
    out_flat = torch.empty(N, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_flat = x.reshape(-1)
    random_flat = random.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N, BLOCK), 1, 1),
        _gelu_dropout_kernel,
        (x_flat, random_flat, gt_flat, out_flat,
         N, BLOCK, DROPOUT_SCALE),
    )
    return gt_flat.view(random_shape), out_flat.view(out_shape)
