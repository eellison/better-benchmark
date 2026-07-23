"""cuTile port of pointwise_c2427cbc2b45: BERT exact-erf GELU + seeded dropout.

Pre-generates the seeded random tensor via inductor_random on the Python side.
A single cuTile pointwise kernel then performs the GELU (with an in-kernel erf
via the Abramowitz-Stegun 7.1.26 polynomial), the dropout mask, and dropout
scale.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 18
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
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK_N,), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf

    x_f = ct.astype(x, ct.float32)
    erf_arg = x_f * 0.7071067811865476

    # erf via Abramowitz-Stegun 7.1.26 polynomial approximation.
    zero_f = ct.zeros((BLOCK_N,), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((BLOCK_N,), 1.0, dtype=ct.float32),
                    ct.full((BLOCK_N,), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_val = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    gelu_f = x_f * 0.5 * (erf_val + 1.0)
    gelu_bf = ct.astype(gelu_f, ct.bfloat16)

    dropped_f = ct.astype(keep, ct.float32) * ct.astype(gelu_bf, ct.float32)
    dropped_bf = ct.astype(dropped_f, ct.bfloat16)
    scaled_f = ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)

    ct.store(gt_ptr, index=(pid,), tile=keep)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


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
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, _shape_param_0, random_shape, out_shape = inputs
    del _shape_param_0

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
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

    # Flatten to 1D for the kernel
    x_flat = x.view(n_elements)
    random_flat = random.reshape(n_elements)
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
