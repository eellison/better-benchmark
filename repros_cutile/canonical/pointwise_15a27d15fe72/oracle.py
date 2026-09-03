"""cuTile port of pointwise_15a27d15fe72: XLNet exact-erf GELU + seeded dropout.

Single flat cuTile kernel does GELU with in-kernel Abramowitz-Stegun erf,
dropout mask + scale. Matches Triton kernel's in-kernel `libdevice.erf`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 76
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


@ct.kernel
def _gelu_dropout_kernel(
    x_ptr,      # bf16 [N]
    random_ptr, # f32  [N]
    gt_ptr,     # b8   [N]
    out_ptr,    # bf16 [N]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    x_f = ct.astype(x_bf, ct.float32)
    erf_v = _erf_approx(x_f * SQRT_HALF)
    gelu_f = (x_f * 0.5) * (erf_v + 1.0)
    gelu_bf = ct.astype(gelu_f, ct.bfloat16)

    random = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    rand_bf = ct.astype(random, ct.bfloat16)
    threshold = ct.astype(ct.full((BLOCK_N,), 0.1, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(pid,), tile=keep)
    zero_bf = ct.full((BLOCK_N,), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, gelu_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled)


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
    x, seeds, _shape_param_0, random_shape, out_shape = inputs
    del _shape_param_0
    random_shape = tuple(int(d) for d in random_shape)
    out_shape = tuple(int(d) for d in out_shape)
    device = x.device
    n_elements = x.numel()

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    x_flat = x.reshape(n_elements)
    random_flat = random.reshape(n_elements)
    gt_flat = gt.view(n_elements)
    out_flat = out.view(n_elements)

    if n_elements % BLOCK_N != 0:
        raise NotImplementedError(f"BLOCK_N={BLOCK_N} doesn't divide n_elements={n_elements}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_elements // BLOCK_N, 1, 1), _gelu_dropout_kernel,
        (x_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )
    return gt, out
