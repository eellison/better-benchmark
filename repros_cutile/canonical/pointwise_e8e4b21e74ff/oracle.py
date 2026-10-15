"""cuTile port of pointwise_e8e4b21e74ff: T5 training ReLU + seeded dropout + le mask.

Pre-generates seeded random via inductor_random outside the kernel, then runs
a single flat cuTile kernel that applies bf16 ReLU with NaN preservation,
seeded dropout mask, dropout scale (1.1111111111111112) and the `<=0` side
mask.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 61
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _relu_dropout_masks_kernel(
    x_ptr,        # bf16 flat [NUMEL]
    random_ptr,   # f32 flat [NUMEL]
    gt_ptr,       # bool flat
    out_ptr,      # bf16 flat
    le_ptr,       # bool flat
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)

    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    random_bf = ct.astype(random_f, ct.bfloat16)
    dropout_p_bf = ct.full((BLOCK_N,), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > dropout_p_bf

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    zero_bf = ct.full((BLOCK_N,), 0.0, dtype=ct.bfloat16)
    # ReLU with NaN preservation: (x > 0) | (x != x)
    relu_mask = (x > zero_bf) | (x != x)
    relu = ct.where(relu_mask, x, zero_bf)
    non_positive = relu <= zero_bf

    dropped = ct.astype(
        ct.astype(keep, ct.float32) * ct.astype(relu, ct.float32),
        ct.bfloat16,
    )
    scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )

    ct.store(gt_ptr, index=(pid,), tile=keep)
    ct.store(out_ptr, index=(pid,), tile=scaled)
    ct.store(le_ptr, index=(pid,), tile=non_positive)


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


@oracle_impl(hardware="B200", point="52dd4c9c", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, _shape_param_0, random_shape_arg, out_shape_arg = inputs

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
    out_base = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=x.device, dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=x.device, dtype=torch.bool,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape, seed, device=x.device
    )

    x_flat = x.contiguous().view(n_elements)
    random_flat = random.contiguous().view(n_elements)
    gt_flat = gt.view(n_elements)
    out_flat = out_base.view(n_elements)
    le_flat = le.view(n_elements)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_N), 1, 1),
        _relu_dropout_masks_kernel,
        (x_flat, random_flat, gt_flat, out_flat, le_flat, BLOCK_N),
    )

    return gt, out_base.view(out_shape), le
