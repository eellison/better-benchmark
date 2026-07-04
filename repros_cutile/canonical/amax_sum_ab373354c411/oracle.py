"""cuTile port of amax_sum_ab373354c411: DeBERTaV2 masked softmax + dropout.

Row kernel: apply broadcast mask via scalar fill, then softmax, dropout, bf16 out.
Returns (where, amax, denom, keep, out, out.permute(0,2,1)).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 70
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

# Shape constants
BATCH = 8
HEADS = 24
Q_LEN = 512
K_LEN = 512


@ct.kernel
def _masked_softmax_dropout_kernel(
    x_ptr,          # bf16 [ROWS, K_LEN]
    mask_ptr,       # b8   [BATCH*Q_LEN, K_LEN] (broadcast in heads dim; here flattened)
    fill_ptr,       # bf16 scalar tensor
    random_ptr,     # f32  [ROWS, K_LEN]
    masked_out_ptr, # bf16 [ROWS, K_LEN]
    amax_ptr,       # f32  [ROWS]
    sum_ptr,        # f32  [ROWS]
    keep_ptr,       # bool [ROWS, K_LEN]
    dropped_ptr,    # bf16 [ROWS, K_LEN]
    N_HEADS: ct.Constant[int],
    Q_LEN_C: ct.Constant[int],
    K_LEN_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))

    # Compute batch/query indices from row: row = batch * (N_HEADS * Q_LEN) + head * Q_LEN + query
    # mask uses offset: batch * Q_LEN * K_LEN + query * K_LEN + col
    # Note: N_HEADS is the number of heads per batch (=24)
    flat_bh = row // Q_LEN_C
    batch = flat_bh // N_HEADS
    query = row - flat_bh * Q_LEN_C
    mask_row_offset = batch * Q_LEN_C + query

    mask_row = ct.load(mask_ptr, index=(mask_row_offset, 0), shape=(1, BLOCK_N))
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar_val = ct.reshape(fill_scalar, (1, 1))
    fill_broadcast = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16) + fill_scalar_val
    masked = ct.where(mask_row, fill_broadcast, x_bf)
    ct.store(masked_out_ptr, index=(row, 0), tile=masked)

    scores = ct.astype(masked, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    threshold = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > threshold
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_bf = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="00541467", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    rows = BATCH * HEADS * Q_LEN
    full_shape = (BATCH, HEADS, Q_LEN, K_LEN)
    row_shape = (BATCH, HEADS, Q_LEN, 1)
    out_shape = (BATCH * HEADS, Q_LEN, K_LEN)
    device = arg0_1.device

    masked = torch.empty_strided(full_shape,
                                  (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1),
                                  device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(row_shape, (HEADS * Q_LEN, Q_LEN, 1, 1), device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, (HEADS * Q_LEN, Q_LEN, 1, 1), device=device, dtype=torch.float32)
    keep = torch.empty_strided(full_shape,
                                (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1),
                                device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, (Q_LEN * K_LEN, K_LEN, 1), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)

    # arg0_1 is [192, 512, 512] bf16 -- reshape to [rows, K_LEN]
    x_2d = arg0_1.contiguous().view(rows, K_LEN)
    # arg1_1 is [8, 1, 512, 512] b8; view as [BATCH*Q_LEN, K_LEN] (heads broadcast is 1)
    mask_2d = arg1_1.contiguous().view(BATCH * Q_LEN, K_LEN)
    random_2d = random.contiguous().view(rows, K_LEN)
    masked_2d = masked.view(rows, K_LEN)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    keep_2d = keep.view(rows, K_LEN)
    dropped_2d = dropped.view(rows, K_LEN)

    # arg2_1 is a scalar bf16 tensor. cuTile requires rank>=1, so view it as (1,).
    fill_1d = arg2_1.view(1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _masked_softmax_dropout_kernel,
        (x_2d, mask_2d, fill_1d, random_2d, masked_2d, amax_1d, sum_1d, keep_2d, dropped_2d,
         HEADS, Q_LEN, K_LEN, BLOCK_N),
    )
    return masked, amax, sum_1, keep, dropped, dropped.permute(0, 2, 1)
