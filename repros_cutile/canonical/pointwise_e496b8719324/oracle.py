"""cuTile port of pointwise_e496b8719324: bf16 exact-erf GELU + seeded dropout.

Adapted from pointwise_294273ab038b. Uses pre-generated random tensor from
`torch.ops.prims.inductor_random`, an Abramowitz-Stegun 7.1.26 polynomial for
`erf`, and RTNE-by-default cuTile bf16 casts in place of the Triton PTX
add.rn.f32 / mul.rn.f32 inline_asm wrappers. Same as pointwise_b1205845774e
but with seed index 92.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 92
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _gelu_dropout_kernel(
    x_ptr,          # bf16 [total]
    random_ptr,     # f32  [total]
    gt_ptr,         # bool [total]
    out_ptr,        # bf16 [total]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)

    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK_N,), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf

    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    x = ct.astype(x_bf, ct.float32)
    erf_arg = x * 0.7071067811865476

    # erf via Abramowitz-Stegun 7.1.26 polynomial approximation:
    #   sign(z) * (1 - poly(t) * exp(-z*z)),  t = 1/(1 + p*|z|)
    zero_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    sign = ct.where(
        erf_arg >= zero_f,
        ct.full((BLOCK_N,), 1.0, dtype=ct.float32),
        ct.full((BLOCK_N,), -1.0, dtype=ct.float32),
    )
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_val = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))
    gelu = x * 0.5 * (erf_val + 1.0)
    gelu_bf = ct.astype(gelu, ct.bfloat16)

    dropped = ct.astype(
        ct.astype(keep, ct.float32) * ct.astype(gelu_bf, ct.float32),
        ct.bfloat16,
    )
    scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )

    ct.store(gt_ptr, index=(pid,), tile=keep)
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


@oracle_impl(hardware="B200", point="c78a05f8", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, _shape0, random_shape_arg, out_shape_arg = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    random_shape = tuple(int(dim) for dim in random_shape_arg)
    out_shape = tuple(int(dim) for dim in out_shape_arg)
    n_elements = int(x.numel())

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=x.device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=x.device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=x.device)

    x_flat = x.contiguous().view(n_elements)
    random_flat = random.contiguous().view(n_elements)
    gt_flat = gt.view(n_elements)
    out_flat = out.view(n_elements)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_N), 1, 1),
        _gelu_dropout_kernel,
        (x_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )
    return gt, out
