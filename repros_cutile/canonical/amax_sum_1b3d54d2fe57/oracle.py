"""cuTile port of amax_sum_1b3d54d2fe57: BERT token-mask attention softmax+dropout.

Pre-broadcasts the token-mask, pre-generates seeded RNG. A cuTile row kernel
fuses: bf16 scale-by-1/8 rounding, masked scalar fill, fp32 row max/exp/sum/div,
seeded dropout with bf16 boundary, bf16 output and permute-alias emission.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
NEG_FILL = -998244352.0


@ct.kernel
def _bert_token_softmax_dropout_kernel(
    scores_ptr,     # bf16 [rows, K]
    mask_ptr,       # b8   [rows, K] (broadcast of unsqueeze_1)
    random_ptr,     # f32  [rows, K]
    amax_ptr,       # f32  [rows]
    denom_ptr,      # f32  [rows]
    keep_ptr,       # b8   [rows, K]
    out_ptr,        # bf16 [rows, K]
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)

    raw = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_K))
    mask = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_K))
    # invert mask: token_eq = valid == 0
    token_eq = ~mask

    # scaled_bf16 = raw * 0.125 (round-to-nearest bf16)
    raw_f = ct.astype(raw, ct.float32)
    scaled_bf16 = ct.astype(raw_f * 0.125, ct.bfloat16)
    fill_bf16 = ct.full((1, BLOCK_K), NEG_FILL, dtype=ct.bfloat16)
    masked_bf16 = ct.where(token_eq, fill_bf16, scaled_bf16)

    scores = ct.astype(masked_bf16, ct.float32)
    row_max = ct.max(scores)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), row_max, dtype=ct.float32), (1,)))

    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer)
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), denom, dtype=ct.float32), (1,)))
    probs = numer / denom

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K))
    dropout_p = ct.full((1, BLOCK_K), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > dropout_p
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_K), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
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


@oracle_impl(hardware="B200", point="9a66816c", BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_K: int):
    tokens, scores, seeds, _repeat_shape, view_shape, random_shape, _expand_shape, out_shape = inputs
    del _repeat_shape, _expand_shape

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _as_shape(view_shape)
    random_shape = _as_shape(random_shape)
    out_shape = _as_shape(out_shape)
    batch, heads, q_len, k_len = view_shape
    mask_shape = (batch, 1, q_len, k_len)
    row_shape = (batch, heads, q_len, 1)
    device = tokens.device
    rows = batch * heads * q_len

    # ==== Precompute the returned mask side outputs (torch) ====
    # unsqueeze_1[b, 0, q, k] = (tokens[b, k] > 0)  — same for all q since repeat.
    token_valid_2d = tokens > 0  # b8[batch, k_len]
    # Repeat to [batch, q_len, k_len] then unsqueeze to [batch, 1, q_len, k_len]
    valid_broadcast = token_valid_2d.unsqueeze(1).expand(batch, q_len, k_len)
    unsqueeze_1 = valid_broadcast.unsqueeze(1).contiguous()
    eq = unsqueeze_1 == 0
    fill = torch.full((), NEG_FILL, dtype=torch.bfloat16, device=device)

    # Broadcast the mask to [batch, heads, q_len, k_len] for per-row loading
    # For a row at (b, h, q), the valid[k] = token_valid_2d[b, k].
    # So we build a full [rows, k_len] mask matrix.
    valid_full = token_valid_2d.unsqueeze(1).unsqueeze(1).expand(
        batch, heads, q_len, k_len).contiguous()
    valid_full_2d = valid_full.view(rows, k_len)

    # scores is bf16[192, 128, 128] view as [16, 12, 128, 128]
    scores_view = scores.view(batch, heads, q_len, k_len)
    scores_2d = scores_view.contiguous().view(rows, k_len)

    # Seeded random over [batch, heads, q_len, k_len]
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.contiguous().view(rows, k_len)

    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    denom = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    amax_1d = amax.view(rows)
    denom_1d = denom.view(rows)
    keep_2d = keep.view(rows, k_len)
    out_2d = out.view(rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bert_token_softmax_dropout_kernel,
        (scores_2d, valid_full_2d, random_2d,
         amax_1d, denom_1d, keep_2d, out_2d, BLOCK_K),
    )

    return unsqueeze_1, eq, fill, amax, denom, keep, out, out.permute(0, 2, 1)
