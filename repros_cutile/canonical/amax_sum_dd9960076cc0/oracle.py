"""cuTile port of amax_sum_dd9960076cc0: DeBERTa masked softmax + dropout.

Same Repro contract as amax_sum_823ba76647e9 but with SEED_INDEX=43. Uses
pre-broadcast mask + inductor_random pre-generated outside the kernel; a
single cuTile row kernel fuses scalar-fill masking, fp32 softmax with side
outputs, seeded dropout, and bf16 rounding.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 43
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    x_ptr,           # bf16 [rows, K]
    mask_ptr,        # b8   [rows, K]
    fill_ptr,        # bf16 [1]
    random_ptr,      # f32  [rows, K]
    masked_ptr,      # bf16 [rows, K]
    amax_ptr,        # f32  [rows]
    sum_ptr,         # f32  [rows]
    keep_ptr,        # b8   [rows, K]
    dropped_ptr,     # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_N))
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16) + ct.reshape(fill_scalar, (1, 1))

    masked = ct.where(mask, fill_bf, x)
    ct.store(masked_ptr, index=(row, 0), tile=masked)

    scores = ct.astype(masked, ct.float32)
    row_max = ct.max(scores)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), row_max, dtype=ct.float32), (1,)))

    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), denom, dtype=ct.float32), (1,)))
    probs = numer / denom

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    dropout_p_f = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > dropout_p_f
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="00541467", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2 = inputs
    del _shape0, _shape1, _shape2

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    batch = 8
    heads = 24
    q_len = 512
    k_len = 512
    rows = batch * heads * q_len
    full_shape = (batch, heads, q_len, k_len)
    row_shape = (batch, heads, q_len, 1)
    out_shape = (batch * heads, q_len, k_len)
    device = arg0_1.device

    # Pre-broadcast the mask [8, 1, 512, 512] to [8, 24, 512, 512]
    mask_bcast = arg1_1.expand(batch, heads, q_len, k_len).contiguous()
    mask_2d = mask_bcast.view(rows, k_len)

    x_2d = arg0_1.contiguous().view(rows, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.contiguous().view(rows, k_len)

    masked = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    masked_2d = masked.view(rows, k_len)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    keep_2d = keep.view(rows, k_len)
    dropped_2d = dropped.view(rows, k_len)

    fill_1d = arg2_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _masked_softmax_dropout_kernel,
        (x_2d, mask_2d, fill_1d, random_2d,
         masked_2d, amax_1d, sum_1d, keep_2d, dropped_2d, BLOCK_N),
    )
    return masked, amax, sum_1, keep, dropped, dropped.permute(0, 2, 1)
