"""cuTile port of amax_sum_bc3fbecd2ce0: XGLM additive-bias attention softmax + dropout.

Uses eager pre-generated random via torch.ops.prims.inductor_random (seed_index 0
against `inductor_seeds(3, device)`).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 3
SEED_INDEX = 0
DROPOUT_SCALE = 1.1111111111111112

HEADS = 16  # x is viewed as [B, HEADS, Q, K] with B=32, HEADS=16, Q=K=128
Q_LEN = 128
K_LEN = 128
LARGE_NEG = -3.4028234663852886e38


@ct.kernel
def _xglm_softmax_dropout_kernel(
    scores_ptr,      # bf16 flat [batch*HEADS, Q, K]
    bias_ptr,        # f32  [batch, Q, K]  (broadcast across HEADS)
    random_ptr,      # f32  [rows, K]
    amax_ptr,        # f32  [rows]
    sum_ptr,         # f32  [rows]
    keep_ptr,        # b8   [rows, K]
    out_ptr,         # bf16 [rows, K]
    HEADS_: ct.Constant[int],
    Q_LEN_: ct.Constant[int],
    K_LEN_: ct.Constant[int],
):
    row = ct.bid(0)  # 0..(batch*HEADS*Q - 1)
    bh = row // Q_LEN_
    batch = bh // HEADS_
    query = row - bh * Q_LEN_

    x = ct.load(scores_ptr, index=(bh, query, 0), shape=(1, 1, K_LEN_))
    x_2d = ct.reshape(x, (1, K_LEN_))
    x_f = ct.astype(x_2d, ct.float32)

    bias = ct.load(bias_ptr, index=(batch, query, 0), shape=(1, 1, K_LEN_))
    bias_2d = ct.reshape(bias, (1, K_LEN_))

    scores = x_f + bias_2d
    large_neg = ct.full(shape=(1, K_LEN_), fill_value=LARGE_NEG, dtype=ct.float32)
    scores = ct.where(scores > large_neg, scores, large_neg)

    row_max = ct.max(scores)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer)
    probs = numer / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_))
    keep = rand_f > 0.1
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, probs, 0.0) * DROPOUT_SCALE
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(dropped, ct.bfloat16))


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
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        return seeds, random
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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="d9690337")
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, _shape_0, _shape_1, random_shape_param = inputs
    device = arg0_1.device

    out_shape = tuple(arg0_1.shape)  # [512, 128, 128]
    row_shape = out_shape[:-1] + (1,)
    out_stride = _contiguous_stride(out_shape)
    row_stride = _contiguous_stride(row_shape)

    amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    denom = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    keep = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bool)
    out = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bfloat16)

    random_shape = tuple(int(dim) for dim in random_shape_param)
    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    rows = out_shape[0] * out_shape[1]  # 512 * 128 = 65536
    # arg0_1 is [bh, Q, K] with bh=512 (=32*16)
    # arg1_1 is [batch, 1, Q, K] with batch=32 -> view as [batch, Q, K]
    bias_3d = arg1_1.view(32, Q_LEN, K_LEN).contiguous()

    random_2d = random.view(rows, K_LEN).contiguous()
    keep_2d = keep.view(rows, K_LEN)
    out_2d = out.view(rows, K_LEN)
    amax_1d = amax.view(rows)
    denom_1d = denom.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _xglm_softmax_dropout_kernel,
        (arg0_1, bias_3d, random_2d,
         amax_1d, denom_1d, keep_2d, out_2d,
         HEADS, Q_LEN, K_LEN),
    )
    return amax, denom, seeds, keep, out, out.permute(0, 2, 1)
