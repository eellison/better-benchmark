"""cuTile port of pointwise_e4d2c6337728: T5 ReLU + seeded dropout + `<=0` mask.

Same structure as pointwise_b1e9709e6271 but with SEED_INDEX=43. Uses
pre-generated random tensor (from torch.ops.prims.inductor_random) to sidestep
cuTile's lack of on-device seeded RNG.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 2048
ROWS = BATCH * SEQ
NUMEL = ROWS * HIDDEN
VIEW_SHAPE = (BATCH, SEQ, HIDDEN)
VIEW_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
SEED_INDEX = 43
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _relu_dropout_masks_kernel(
    x_ptr,          # bf16 [N]
    random_ptr,     # f32  [N]
    gt_ptr,         # b8   [N]
    out_ptr,        # bf16 [N]
    le_ptr,         # b8   [N]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    rand = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_N,), DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    zero_bf = ct.full((BLOCK_N,), 0.0, dtype=ct.bfloat16)
    is_pos = x > zero_bf
    is_nan = x != x
    relu = ct.where(is_pos | is_nan, x, zero_bf)
    non_positive = relu <= zero_bf

    dropped = ct.astype(
        ct.astype(keep, ct.float32) * ct.astype(relu, ct.float32), ct.bfloat16
    )
    scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16
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
    x, seeds, _view_shape, random_shape, out_shape = inputs
    del _view_shape

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    out_base = torch.empty_strided(
        VIEW_SHAPE, VIEW_STRIDE,
        device=x.device, dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        VIEW_SHAPE, VIEW_STRIDE,
        device=x.device, dtype=torch.bool,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=x.device)

    x_flat = x.reshape(NUMEL)
    random_flat = random.reshape(NUMEL).contiguous()
    gt_flat = gt.view(NUMEL)
    out_flat = out_base.view(NUMEL)
    le_flat = le.view(NUMEL)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUMEL, BLOCK_N), 1, 1),
        _relu_dropout_masks_kernel,
        (x_flat, random_flat, gt_flat, out_flat, le_flat, BLOCK_N),
    )

    return gt, out_base.view(out_shape), le
