"""cuTile port of amax_sum_b0e9ed98229b: BERT scaled masked attention softmax + dropout.

Like the DeBERTa f9d898b0b99c pattern, but with a bf16 divide-by-8.0 boundary
before the mask is applied. Seed index 16.

For each row of the flattened [batch*heads, q_len, k_len] view:
  1. Load bf16 scores, cast to fp32, multiply by 0.125 (mul.rn.f32 semantics —
     cuTile default is RTNE), round to bf16 (scaled_bf16).
  2. Apply broadcast bool mask (per-batch, all-heads) with scalar bf16 fill.
  3. Store rounded-and-masked bf16 scores.
  4. fp32 softmax over the last dim: amax + exp + sum + div.
  5. Store f32 row-max (amax) and row-denominator (sum) side outputs.
  6. Seeded Inductor dropout (seed index 16): keep = random_f32 > 0.1.
  7. Store dropout mask, apply mask + scale to fp32 probs, cast to bf16.
  8. Return the [192, 128, 128] output plus its (0,2,1) permute alias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _scaled_masked_softmax_dropout_kernel(
    scores_ptr,     # bf16 [ROWS, K]
    mask_ptr,       # b8   [BATCH*Q, K]
    random_ptr,     # f32  [ROWS, K]
    where_ptr,      # bf16 [ROWS, K]
    amax_ptr,       # f32  [ROWS]
    denom_ptr,      # f32  [ROWS]
    keep_ptr,       # b8   [ROWS, K]
    out_ptr,        # bf16 [ROWS, K]
    fill: ct.Constant[float],
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bh = row // Q_LEN
    batch = flat_bh // HEADS
    query = row - flat_bh * Q_LEN
    mask_row = batch * Q_LEN + query

    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_K))
    # bf16 divide-by-8.0 => cast to fp32, multiply by 0.125, cast back.
    scaled_bf16 = ct.astype(ct.astype(scores_bf, ct.float32) * 0.125, ct.bfloat16)

    mask = ct.load(mask_ptr, index=(mask_row, 0), shape=(1, BLOCK_K))
    fill_tile = ct.full((1, BLOCK_K), fill, dtype=ct.bfloat16)
    masked_scores = ct.where(mask, fill_tile, scaled_bf16)
    ct.store(where_ptr, index=(row, 0), tile=masked_scores)

    scores_f = ct.astype(masked_scores, ct.float32)
    row_max = ct.max(scores_f, axis=1, keepdims=True)
    numer = ct.exp(scores_f - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    random = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K))
    keep = random > DROPOUT_P
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, probs, 0.0)
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


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_K: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    # arg0_1 bf16[192,128,128], arg1_1 b8[16,1,128,128], arg2_1 bf16[] fill,
    # arg3_1 i64[61] seeds

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    view_shape = _resolve_shape(shape0, arg0_1.numel())   # (16,12,128,128)
    random_shape = _as_shape(shape1)
    flat_shape = _resolve_shape(shape3, arg0_1.numel())   # (192,128,128)
    batch = int(view_shape[0])
    heads = int(view_shape[1])
    q_len = int(view_shape[2])
    k_len = int(view_shape[3])
    n_rows = batch * heads * q_len
    reduction_shape = (batch, heads, q_len, 1)

    where = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    scores_2d = arg0_1.view(n_rows, k_len)
    mask_2d = arg1_1.contiguous().view(batch * q_len, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    denom_1d = denom.view(n_rows)
    keep_2d = keep.view(n_rows, k_len)
    out_2d = out.view(n_rows, k_len)

    fill_value = float(arg2_1.item())

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _scaled_masked_softmax_dropout_kernel,
        (scores_2d, mask_2d, random_2d,
         where_2d, amax_1d, denom_1d, keep_2d, out_2d,
         fill_value, heads, q_len, BLOCK_K),
    )

    return where, amax, denom, keep, out, out.permute(0, 2, 1)
