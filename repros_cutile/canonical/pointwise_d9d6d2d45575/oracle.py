"""cuTile port of pointwise_d9d6d2d45575: BERT/XLNet exact-erf GELU + seeded dropout.

Pre-generates the seeded random tensor via inductor_random on the Python side,
then runs one flat cuTile pointwise kernel that: applies exact-erf GELU in f32,
rounds to bf16, generates dropout mask from f32-to-bf16-cast random comparison
against 0.1, and emits (gt mask, scaled bf16 output).

Uses Abramowitz-Stegun 7.1.26 polynomial for erf (cuTile lacks ct.erf).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 28
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _gelu_dropout_kernel(
    x_ptr,          # bf16 [total]
    random_ptr,     # f32 [total]
    gt_ptr,         # bool [total]
    out_ptr,        # bf16 [total]
    n_elements: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_N_ + ct.arange(BLOCK_N_, dtype=ct.int32)
    mask = ct.reshape(offsets < n_elements, (BLOCK_N_,))

    random_f = ct.load(
        random_ptr, index=(pid,), shape=(BLOCK_N_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random_bf16 = ct.astype(random_f, ct.bfloat16)
    threshold = ct.astype(
        ct.full((BLOCK_N_,), DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf16 > threshold

    x = ct.load(
        x_ptr, index=(pid,), shape=(BLOCK_N_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f32 = ct.astype(x, ct.float32)

    # Exact erf GELU: 0.5 * x * (erf(x / sqrt(2)) + 1)
    erf_arg = x_f32 * 0.7071067811865476
    erf_val = _erf_approx(erf_arg)
    gelu = 0.5 * x_f32 * (erf_val + 1.0)
    gelu_bf16 = ct.astype(gelu, ct.bfloat16)

    dropped = ct.where(keep, ct.astype(gelu_bf16, ct.float32), 0.0)
    dropped_bf16 = ct.astype(dropped, ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)

    keep_masked = ct.where(mask, keep, ct.full((BLOCK_N_,), False, dtype=ct.bool_))
    scaled_masked = ct.where(mask, scaled_bf16, ct.zeros((BLOCK_N_,), dtype=ct.bfloat16))

    ct.store(gt_ptr, index=(pid,), tile=keep_masked)
    ct.store(out_ptr, index=(pid,), tile=scaled_masked)


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


@oracle_impl(hardware="B200", point="f9ed8dd6", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="c78a05f8", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, _shape_param_0, random_shape, out_shape = inputs
    del _shape_param_0

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    device = x.device
    n_elements = x.numel()

    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(-1).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(n_elements, BLOCK_N), 1, 1), _gelu_dropout_kernel,
        (x.view(-1), random_flat, gt.view(-1), out.view(-1), n_elements, BLOCK_N),
    )
    return gt, out
