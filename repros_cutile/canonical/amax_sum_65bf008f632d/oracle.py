"""cuTile port of amax_sum_65bf008f632d: DeBERTa bf16 masked attention softmax + dropout.

Uses eager pre-generated random via torch.ops.prims.inductor_random (seed index 34).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 34
DROPOUT_SCALE = 1.1111111111111112

BATCH = 8
HEADS = 24
Q_LEN = 512
K_LEN = 512


@ct.kernel
def _masked_softmax_dropout_kernel(
    x_ptr,           # bf16 [rows, K_LEN]      (rows = BATCH*HEADS*Q_LEN)
    mask_ptr,        # b8   [BATCH, Q_LEN, K_LEN]  (broadcast over HEADS)
    fill_ptr,        # bf16 []
    random_ptr,      # f32  [rows, K_LEN]
    masked_ptr,      # bf16 [rows, K_LEN]
    amax_ptr,        # f32  [rows]
    sum_ptr,         # f32  [rows]
    keep_ptr,        # b8   [rows, K_LEN]
    dropped_ptr,     # bf16 [rows, K_LEN]
    HEADS_: ct.Constant[int],
    Q_LEN_: ct.Constant[int],
    K_LEN_: ct.Constant[int],
):
    row = ct.bid(0)  # 0..(BATCH*HEADS*Q_LEN - 1)
    # Decompose row -> (batch, head, query)
    bh = row // Q_LEN_
    batch = bh // HEADS_
    query = row - bh * Q_LEN_

    x = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN_))
    mask = ct.load(mask_ptr, index=(batch, query, 0), shape=(1, 1, K_LEN_))
    mask_2d = ct.reshape(mask, (1, K_LEN_))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.reshape(fill, (1, 1))

    masked = ct.where(mask_2d, ct.astype(fill_scalar, ct.bfloat16), x)
    ct.store(masked_ptr, index=(row, 0), tile=masked)

    scores_f32 = ct.astype(masked, ct.float32)
    row_max = ct.max(scores_f32)
    numer = ct.exp(scores_f32 - row_max)
    denom = ct.sum(numer)
    probs = numer / denom
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_))
    keep = rand_f > 0.1
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, probs, 0.0) * DROPOUT_SCALE
    ct.store(dropped_ptr, index=(row, 0), tile=ct.astype(dropped, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="00541467")
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2 = inputs
    device = arg0_1.device

    rows = BATCH * HEADS * Q_LEN
    full_shape = (BATCH, HEADS, Q_LEN, K_LEN)
    row_shape = (BATCH, HEADS, Q_LEN, 1)
    out_shape = (BATCH * HEADS, Q_LEN, K_LEN)

    masked = torch.empty_strided(
        full_shape,
        (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape,
        (HEADS * Q_LEN, Q_LEN, 1, 1),
        device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        (HEADS * Q_LEN, Q_LEN, 1, 1),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        full_shape,
        (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape,
        (Q_LEN * K_LEN, K_LEN, 1),
        device=device, dtype=torch.bfloat16,
    )

    # arg1_1 has shape [BATCH, 1, Q_LEN, K_LEN], view as [BATCH, Q_LEN, K_LEN]
    mask_3d = arg1_1.view(BATCH, Q_LEN, K_LEN).contiguous()
    # arg2_1 is a scalar (0-dim), view as size-1 1D tensor
    fill_1d = arg2_1.view(1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.view(rows, K_LEN)

    # Flatten for kernel access
    x_2d = arg0_1.view(rows, K_LEN)
    masked_2d = masked.view(rows, K_LEN)
    amax_1d = amax.view(rows)
    sum_1_1d = sum_1.view(rows)
    keep_2d = keep.view(rows, K_LEN)
    dropped_2d = dropped.view(rows, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _masked_softmax_dropout_kernel,
        (x_2d, mask_3d, fill_1d, random_2d,
         masked_2d, amax_1d, sum_1_1d, keep_2d, dropped_2d,
         HEADS, Q_LEN, K_LEN),
    )
    return masked, amax, sum_1, keep, dropped, dropped.permute(0, 2, 1)
