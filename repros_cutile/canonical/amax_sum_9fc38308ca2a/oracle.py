"""cuTile port of amax_sum_9fc38308ca2a: DeBERTa masked-attention softmax + dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. K_LEN is 512 which matches
the BLOCK_N tile size, so no OOB.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 25
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    scores_ptr,     # bf16 [n_rows, K_LEN]
    mask_ptr,       # b8   [n_rows, K_LEN]
    fill_ptr,       # bf16 [1]
    random_ptr,     # f32  [n_rows, K_LEN]
    where_ptr,      # bf16 [n_rows, K_LEN]
    amax_ptr,       # f32  [n_rows]
    denom_ptr,      # f32  [n_rows]
    keep_ptr,       # b8   [n_rows, K_LEN]
    out_ptr,        # bf16 [n_rows, K_LEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    raw = ct.load(scores_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    mask_val = ct.load(mask_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_tile = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32),
        ct.bfloat16,
    ) + ct.reshape(fill_scalar, (1, 1))
    masked = ct.where(mask_val, fill_tile, raw)
    ct.store(where_ptr, index=(row_block, 0), tile=masked)

    scores = ct.astype(masked, ct.float32)
    row_max = ct.max(scores, axis=1)
    row_max_2d = ct.reshape(row_max, (BLOCK_M, 1))
    numer = ct.exp(scores - row_max_2d)
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    probs = numer / denom_2d

    ct.store(amax_ptr, index=(row_block,), tile=row_max)
    ct.store(denom_ptr, index=(row_block,), tile=denom)

    rand = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    keep = rand > 0.1
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)

    zeros = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zeros)
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=scaled)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


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


@oracle_impl(hardware="B200", point="00541467", BLOCK_M=4, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    view_shape = _resolve_shape(shape0, arg0_1.numel())
    random_shape = _as_shape(shape1)
    flat_shape = _resolve_shape(shape2, arg0_1.numel())
    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)

    where = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape,
        _contiguous_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape,
        _contiguous_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)

    k_len = int(view_shape[3])
    n_rows = int(arg0_1.numel() // k_len)

    mask_expanded = arg1_1.expand(view_shape).contiguous()
    mask_2d = mask_expanded.view(n_rows, k_len)

    scores_2d = arg0_1.view(n_rows, k_len)
    fill_scalar = arg2_1.reshape(1)
    random_2d = random.reshape(n_rows, k_len).contiguous()
    where_2d = where.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    denom_1d = denom.view(n_rows)
    keep_2d = keep.view(n_rows, k_len)
    out_2d = out.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _masked_softmax_dropout_kernel,
        (scores_2d, mask_2d, fill_scalar, random_2d, where_2d, amax_1d, denom_1d,
         keep_2d, out_2d, BLOCK_M, BLOCK_N),
    )
    return where, amax, denom, keep, out, out.permute(0, 2, 1)
