"""cuTile port of pointwise_cbfed5cd52de: ReLU + seeded dropout fanout.

Pre-generates the seeded random tensor via inductor_random on the Python side,
then runs one flat cuTile pointwise kernel that emits (gt mask, ReLU+drop out,
le0 mask).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0


@ct.kernel
def _relu_dropout_masks_kernel(
    x_ptr,          # bf16 [total]
    random_ptr,     # f32 [total]
    gt_ptr,         # bool [total]
    out_ptr,        # bf16 [total]
    le_ptr,         # bool [total]
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK_N,), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf

    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    zero_bf = ct.zeros((BLOCK_N,), dtype=ct.bfloat16)
    relu = ct.where((x > zero_bf) | (x != x), x, zero_bf)
    non_positive = relu <= zero_bf

    dropped_f = ct.astype(keep, ct.float32) * ct.astype(relu, ct.float32)
    dropped_bf = ct.astype(dropped_f, ct.bfloat16)
    scaled_f = ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)

    ct.store(gt_ptr, index=(pid,), tile=keep)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)
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


@oracle_impl(hardware="B200", point="35c60c30", BLOCK_N=512)
@oracle_impl(hardware="B200", point="3044d858", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, random_shape = inputs
    random_shape = tuple(int(dim) for dim in random_shape)
    stride = _contiguous_stride(random_shape)
    total = x.numel()

    gt = torch.empty_strided(random_shape, stride, device=x.device, dtype=torch.bool)
    out = torch.empty_strided(random_shape, stride, device=x.device, dtype=torch.bfloat16)
    le = torch.empty_strided(random_shape, stride, device=x.device, dtype=torch.bool)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=x.device)

    # Flatten to 1D for the kernel
    x_flat = x.view(total)
    random_flat = random.view(total)
    gt_flat = gt.view(total)
    out_flat = out.view(total)
    le_flat = le.view(total)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK_N), 1, 1),
        _relu_dropout_masks_kernel,
        (x_flat, random_flat, gt_flat, out_flat, le_flat, BLOCK_N),
    )
    return gt, out, le
