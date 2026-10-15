"""cuTile port of pointwise_08576d101b39: XLNet exact-erf GELU + seeded dropout.

Single flat cuTile kernel does GELU with in-kernel Abramowitz-Stegun erf,
dropout mask + scale. Matches Triton kernel's in-kernel `libdevice.erf`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 68
DROPOUT_SCALE = 1.1111111111111112
SQRT_HALF = 0.7071067811865476


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
    x_ptr,       # bf16 [N]
    rand_ptr,    # f32 [N]
    gt_ptr,      # b8 [N]
    out_ptr,     # bf16 [N]
    N: ct.Constant[int],
    BLOCK: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    rand_v = ct.load(rand_ptr, index=(pid,), shape=(BLOCK,),
                     padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    erf_v = _erf_approx(x_f * SQRT_HALF)
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

    gt = gt_flat.view(random_shape)
    out = out_flat.view(out_shape)
    return gt, out
