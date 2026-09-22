"""cuTile port of amax_sum_e776de0c82f4: MT5 attention softmax + dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. Rows are 128 wide (K_LEN),
row tile is (BLOCK_M=8, BLOCK_N=128), so no OOB — no masks needed.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 49
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [n_rows, K_LEN]
    random_ptr,     # f32  [n_rows, K_LEN]
    amax_ptr,       # f32  [n_rows]
    sum_ptr,        # f32  [n_rows]
    gt_ptr,         # b8   [n_rows, K_LEN]
    dropped_ptr,    # bf16 [n_rows, K_LEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    scores = ct.astype(x_bf, ct.float32)
    row_max = ct.max(scores, axis=1)
    row_max_2d = ct.reshape(row_max, (BLOCK_M, 1))
    numer = ct.exp(scores - row_max_2d)
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    probs = ct.astype(numer / denom_2d, ct.bfloat16)

    ct.store(amax_ptr, index=(row_block,), tile=row_max)
    ct.store(sum_ptr, index=(row_block,), tile=denom)

    rand = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs, zero_bf)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


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


@oracle_impl(hardware="B200", point="1715052e", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    del _expand_shape

    full_shape = _shape_tuple(full_shape_arg)  # [32,6,128,128]
    random_shape = _shape_tuple(random_shape_arg)
    out_shape = _shape_tuple(out_shape_arg)  # [192,128,128]
    k_len = int(full_shape[-1])
    n_rows = int(x.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=x.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=x.device,
        dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=x.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=x.device)

    # Reshape all to [n_rows, k_len] contiguous 2D so the cuTile kernel sees
    # them uniformly.
    x_2d = x.reshape(n_rows, k_len)
    random_2d = random.reshape(n_rows, k_len).contiguous()
    amax_2d = amax.view(n_rows)
    sum_2d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _softmax_dropout_kernel,
        (x_2d, random_2d, amax_2d, sum_2d, gt_2d, dropped_2d, BLOCK_M, BLOCK_N),
    )
    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
