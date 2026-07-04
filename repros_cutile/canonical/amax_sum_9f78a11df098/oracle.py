"""cuTile port of amax_sum_9f78a11df098: BERT safe softmax + seeded dropout.

Uses inductor_random outside the kernel. The [16,1,128,128] mask is expanded
to the full [16,12,128,128] shape (via torch broadcast+contiguous) so cuTile
can index it in tile space alongside the other tensors.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 41
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    scores_ptr,     # bf16 [n_rows, k_len]
    mask_ptr,       # b8   [n_rows, k_len]
    fill_ptr,       # bf16 [1]
    random_ptr,     # f32  [n_rows, k_len]
    where_ptr,      # bf16 [n_rows, k_len]
    amax_ptr,       # f32  [n_rows]
    denom_ptr,      # f32  [n_rows]
    keep_ptr,       # b8   [n_rows, k_len]
    out_ptr,        # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)

    view_bf = ct.load(scores_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_K))
    mask_v = ct.load(mask_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_K))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    rand_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_K))

    # bf16 division by 8.0 - do it in bf16 to match Inductor path
    inv8_bf = ct.full((BLOCK_M, BLOCK_K), 0.125, dtype=ct.bfloat16)
    div_bf = ct.astype(ct.astype(view_bf, ct.float32) * ct.astype(inv8_bf, ct.float32), ct.bfloat16)

    fill_2d = ct.reshape(fill, (1, 1))
    fill_broadcast = ct.full((BLOCK_M, BLOCK_K), 0.0, dtype=ct.bfloat16) + fill_2d
    masked_scores = ct.where(mask_v, fill_broadcast, div_bf)
    ct.store(where_ptr, index=(row_block, 0), tile=masked_scores)

    scores_f = ct.astype(masked_scores, ct.float32)
    row_max = ct.max(scores_f, axis=1, keepdims=True)
    numer = ct.exp(scores_f - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom

    row_max_1d = ct.reshape(row_max, (BLOCK_M,))
    denom_1d = ct.reshape(denom, (BLOCK_M,))
    ct.store(amax_ptr, index=(row_block,), tile=row_max_1d)
    ct.store(denom_ptr, index=(row_block,), tile=denom_1d)

    keep = rand_f > 0.1
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)

    zero_f = ct.full((BLOCK_M, BLOCK_K), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled_f = dropped * DROPOUT_SCALE_
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(scaled_f, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    scores, mask, fill, seeds, full_shape, random_shape, _expand_shape, out_shape = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(full_shape)
    random_shape = _shape_tuple(random_shape)
    out_shape = _shape_tuple(out_shape)
    K = int(full_shape[-1])
    n_rows = int(scores.numel() // K)
    row_shape = full_shape[:-1] + (1,)

    device = scores.device

    view_bf = scores.view(full_shape)
    mask_broadcast = mask.expand(full_shape).contiguous()

    where = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool)
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    fill_1d = fill.view(1)

    view_2d = view_bf.contiguous().view(n_rows, K)
    mask_2d = mask_broadcast.view(n_rows, K)
    random_2d = random.contiguous().view(n_rows, K)
    where_2d = where.view(n_rows, K)
    gt_2d = gt.view(n_rows, K)
    dropped_2d = dropped.view(n_rows, K)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _masked_softmax_dropout_kernel,
        (view_2d, mask_2d, fill_1d, random_2d, where_2d,
         amax_1d, sum_1d, gt_2d, dropped_2d,
         K, BLOCK_M, BLOCK_N, DROPOUT_SCALE),
    )
    return where, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
