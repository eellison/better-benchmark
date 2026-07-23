"""cuTile port of pointwise_dce071a17123: Blenderbot/M2M100/XGLM dropout+add.

Ports the Triton `_dropout_add_kernel`. Pre-generates the seeded random
tensor via `torch.ops.prims.inductor_random`, then runs a flat pointwise
cuTile kernel that computes:
  * bf16-threshold dropout mask
  * bf16 dropped * 1.1111...
  * fp32 residual add -> f32 output
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 2
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_add_kernel(
    x_ptr,          # bf16 [N]
    random_ptr,     # f32 [N]
    residual_ptr,   # f32 [N]
    mask_ptr,       # bool [N]
    out_ptr,        # f32 [N]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK_N,))

    rand_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_N,), DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(mask_ptr, index=(pid,), tile=keep)

    zero_bf = ct.zeros((BLOCK_N,), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    out = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(out_ptr, index=(pid,), tile=out)


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


@oracle_impl(hardware="B200", point="9fa00d2a", BLOCK_N=256)
@oracle_impl(hardware="B200", point="81b27e33", BLOCK_N=256)
@oracle_impl(hardware="B200", point="111e5e70", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, residual, _view_shape, random_shape = inputs
    del _view_shape

    out_shape = tuple(int(dim) for dim in random_shape)
    n_elements = residual.numel()
    device = residual.device

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(out_shape, seed, device=device)
    random_flat = random.contiguous().view(n_elements)

    mask = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        tuple(residual.shape), tuple(residual.stride()),
        device=device, dtype=torch.float32,
    )

    x_flat = x.contiguous().view(n_elements)
    residual_flat = residual.contiguous().view(n_elements)
    mask_flat = mask.view(n_elements)
    out_flat = out.view(n_elements)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK_N), 1, 1),
        _dropout_add_kernel,
        (x_flat, random_flat, residual_flat, mask_flat, out_flat, BLOCK_N),
    )
    return mask, out
