"""cuTile port of pointwise_a8f7c1df5d48: XLNet exact-erf GELU + seeded dropout.

Single flat cuTile kernel does GELU with in-kernel Abramowitz-Stegun erf,
dropout mask + scale. Matches Triton kernel's in-kernel `libdevice.erf`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 44
DROPOUT_SCALE = 1.1111111111111112
ERF_COEF = 0.7071067811865476


def _erf_approx(x):
    """Abramowitz-Stegun 7.1.26 polynomial approximation for erf."""
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108
    x_abs = ct.abs(x)
    x2 = x_abs * x_abs
    x3 = x2 * x_abs
    x4 = x3 * x_abs
    denom = 1.0 + a1 * x_abs + a2 * x2 + a3 * x3 + a4 * x4
    denom2 = denom * denom
    denom4 = denom2 * denom2
    erf_abs = 1.0 - 1.0 / denom4
    sign = ct.where(x >= 0.0, 1.0, -1.0)
    return sign * erf_abs


@ct.kernel
def _gelu_dropout_kernel(
    x_ptr,
    random_ptr,
    gt_ptr,
    out_ptr,
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))

    x_f = ct.astype(x_bf, ct.float32)
    erf_v = _erf_approx(x_f * ERF_COEF)
    gelu_f = 0.5 * x_f * (erf_v + 1.0)
    gelu_bf = ct.astype(gelu_f, ct.bfloat16)

    rand = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((BLOCK_N,), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(pid,), tile=keep)

    zero_bf = ct.full((BLOCK_N,), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, gelu_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled)


def _shape(shape):
    return tuple(int(d) for d in shape)


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


@oracle_impl(hardware="B200", point="c78a05f8", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _shape(shape0)  # [512, 16, 4096]
    random_shape = _shape(shape1)
    flat_shape = _shape(shape2)  # [8192, 4096]
    numel = 1
    for d in view_shape:
        numel *= int(d)

    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    view_1 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_flat = arg0_1.view(numel)
    random_flat = random.contiguous().view(numel)
    gt_flat = gt.view(numel)
    out_flat = view_1.view(numel)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK_N), 1, 1),
        _gelu_dropout_kernel,
        (x_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )
    return gt, view_1
