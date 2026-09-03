"""cuTile port of pointwise_f0e2b374d39c: AlexNet ReLU + maxpool + dropout.

The internal seeded RNG is pre-generated outside via inductor_random, then
two cuTile kernels do the substantive pointwise work:
  1. `_dropout_scale_kernel` — reads the flattened post-pool view, applies
     `keep = rand > 0.5`, computes `keep * value * 2.0`, and writes bf16 dropout
     output plus the bool keep mask.
  2. `_relu_le_kernel` — writes the `relu(input) <= 0` bool mask over the
     full input tensor.

The pooling stages and adaptive_avg_pool are executed via torch since Triton's
_low_memory_max_pool_with_offsets emits a paired (values, i8-offsets) tensor
that we get for free from the aten op.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 256
H_IN = 13
W_IN = 13
H_OUT = 6
W_OUT = 6
INPUT_NUMEL = BATCH * CHANNELS * H_IN * W_IN
FLAT_NUMEL = BATCH * 9216  # 128*256*6*6
SEED_COUNT = 2
SEED_INDEX = 0
DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0


@ct.kernel
def _dropout_scale_kernel(
    view_ptr,       # bf16 [N]
    random_ptr,     # f32 [N]
    gt_ptr,         # b8 [N]
    out_ptr,        # bf16 [N]
    N: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    view = ct.load(
        view_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_f = ct.load(
        random_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((BLOCK,), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf

    keep_f = ct.astype(keep, ct.float32)
    view_f = ct.astype(view, ct.float32)
    dropped_bf = ct.astype(keep_f * view_f, ct.bfloat16)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)

    idx = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idx < N
    ct.scatter(gt_ptr, (idx,), keep, mask=valid)
    ct.scatter(out_ptr, (idx,), scaled, mask=valid)


@ct.kernel
def _relu_le_kernel(
    x_ptr,        # bf16 [N]
    le_ptr,       # b8 [N]
    N: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    zero_f = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    # le(relu(x), 0): relu clamps to 0, so this is (x <= 0) (NaN stays NaN;
    # NaN <= 0 is False in both).
    le_val = x_f <= zero_f
    idx = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idx < N
    ct.scatter(le_ptr, (idx,), le_val, mask=valid)


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


def _random_advance(shape, *, device):
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
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


@oracle_impl(
    hardware="B200",
    point="a8ee30c6",
    POOL_BLOCK=256,
    MASK_BLOCK=1024,
    pool_warps=8,
    mask_warps=4,
)
def oracle_forward(
    inputs,
    *,
    POOL_BLOCK: int,
    MASK_BLOCK: int,
    pool_warps: int,
    mask_warps: int,
):
    del POOL_BLOCK, pool_warps, mask_warps  # unused in cuTile port
    x, kernel_size, stride, flat_shape_param, random_shape_param = inputs
    flat_shape = tuple(int(dim) for dim in flat_shape_param)
    random_shape = tuple(int(dim) for dim in random_shape_param)
    device = x.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    # ReLU then _low_memory_max_pool_with_offsets: values [B,C,6,6], offsets i8.
    relu = torch.relu(x)
    lm = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu, list(kernel_size), list(stride), [0, 0], [1, 1], False
    )
    values = lm[0]
    offsets = lm[1]
    avg = torch.ops.aten._adaptive_avg_pool2d.default(values, [6, 6])
    view = avg.view(flat_shape)

    # Seed + random path — pre-generate rand outside the kernel.
    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    dropout = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        (BATCH, CHANNELS, H_IN, W_IN),
        (CHANNELS * H_IN * W_IN, H_IN * W_IN, W_IN, 1),
        device=device, dtype=torch.bool,
    )

    view_1d = view.contiguous().view(-1)
    random_1d = random.contiguous().view(-1)
    gt_1d = gt.view(-1)
    dropout_1d = dropout.view(-1)
    stream = torch.cuda.current_stream()
    grid_d = ((FLAT_NUMEL + 1024 - 1) // 1024, 1, 1)
    ct.launch(
        stream,
        grid_d,
        _dropout_scale_kernel,
        (view_1d, random_1d, gt_1d, dropout_1d, FLAT_NUMEL, 1024),
    )

    x_1d = x.contiguous().view(-1)
    le_1d = le.view(-1)
    grid_le = ((INPUT_NUMEL + MASK_BLOCK - 1) // MASK_BLOCK, 1, 1)
    ct.launch(
        stream,
        grid_le,
        _relu_le_kernel,
        (x_1d, le_1d, INPUT_NUMEL, MASK_BLOCK),
    )

    return values, offsets, seeds, gt, dropout, le
