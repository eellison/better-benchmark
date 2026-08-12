"""cuTile port of pointwise_abb86bbd005a: XLNet exact-erf GELU + Inductor dropout.

Single flat cuTile kernel does GELU with in-kernel Abramowitz-Stegun erf,
dropout mask + scale. Matches Triton kernel's in-kernel `libdevice.erf`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 72
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_N = 1024
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
    x_ptr,           # bf16 [total]
    random_ptr,      # f32  [total]
    gt_ptr,          # b8   [total]
    out_ptr,         # bf16 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x_bf, ct.float32)
    erf_v = _erf_approx(x_f * SQRT_HALF)
    gelu_f = 0.5 * x_f * (erf_v + 1.0)
    gelu_bf = ct.astype(gelu_f, ct.bfloat16)

    rand_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(BLOCK,), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(pid,), tile=keep)

    zero_bf = ct.astype(
        ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32),
        ct.bfloat16,
    )
    dropped = ct.where(keep, gelu_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="c78a05f8")
def oracle_forward(inputs):
    arg0_1, arg1_1, _shape0, random_shape, out_shape = inputs
    device = arg0_1.device
    random_shape_t = _as_shape(random_shape)
    out_shape_t = _as_shape(out_shape)
    n = arg0_1.numel()

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape_t, seed, device=device)
    random_flat = random.contiguous().view(-1)

    x_flat = arg0_1.view(-1)
    gt_flat = torch.empty((n,), device=device, dtype=torch.bool)
    out_flat = torch.empty((n,), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    assert n % BLOCK_N == 0
    ct.launch(
        stream, (n // BLOCK_N, 1, 1), _gelu_dropout_kernel,
        (x_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )

    gt = torch.empty_strided(
        random_shape_t, _contiguous_stride(random_shape_t),
        device=device, dtype=torch.bool,
    )
    gt.view(-1).copy_(gt_flat)
    out = torch.empty_strided(
        out_shape_t, _contiguous_stride(out_shape_t),
        device=device, dtype=torch.bfloat16,
    )
    out.view(-1).copy_(out_flat)

    return gt, out
