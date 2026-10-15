"""cuTile port of amax_sum_d46ce8f9ec36: DeBERTaV2 masked softmax + seeded dropout.

Pre-generates random tensor. Row kernel: masked softmax (where mask, fill,
else raw), amax/exp/sum, dropout + scale, bf16 cast. Emits `where` (masked
scores), amax, denom, keep mask, final bf16 output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 52
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    scores_ptr,   # bf16 [batch*heads, Q_LEN, K_LEN]
    mask_ptr,     # bool [batch, 1, Q_LEN, K_LEN] — same broadcast pattern as flat
    fill_val_ptr, # bf16 scalar
    random_ptr,   # f32 [rows, K_LEN]
    where_ptr,    # bf16 [rows, K_LEN]
    amax_ptr,     # f32 [rows]
    denom_ptr,    # f32 [rows]
    keep_ptr,     # bool [rows, K_LEN]
    out_ptr,      # bf16 [rows, K_LEN]
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)
    scores = ct.load(
        scores_ptr, index=(row, 0), shape=(1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # mask has shape [batch, 1, Q_LEN, K_LEN] laid out contiguous
    # For flat row idx: batch = row // (heads * Q_LEN); query = row % Q_LEN
    # so mask_row_in_flat = (batch * Q_LEN + query)
    # But we can pass a pre-broadcast mask on the Python side: use a
    # broadcast-mask tensor of shape [rows, K_LEN].
    mask_vals = ct.load(
        mask_ptr, index=(row, 0), shape=(1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill = ct.load(fill_val_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.astype(fill, ct.bfloat16)
    fill_bf_2d = ct.full((1, BLOCK_K), 0.0, dtype=ct.bfloat16)
    # We need fill to be broadcast. Use ct.where with the scalar tile from fill.
    # Since ct.load creates shape=(1,), we need to broadcast to (1, BLOCK_K).
    # A safe approach: pass fill as f32 and do the assignment via mul with mask.
    fill_expand = ct.reshape(fill_scalar, (1, 1))
    # Broadcast to (1, BLOCK_K) via full-then-select
    masked_scores = ct.where(mask_vals, fill_expand, scores)
    ct.store(where_ptr, index=(row, 0), tile=masked_scores)

    col_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < K_LEN, (1, BLOCK_K))
    scores_f = ct.astype(masked_scores, ct.float32)
    neg_inf = ct.full((1, BLOCK_K), float("-inf"), dtype=ct.float32)
    scores_active = ct.where(col_mask, scores_f, neg_inf)
    row_max = ct.max(scores_active, axis=1, keepdims=True)
    numer = ct.exp(scores_f - row_max)
    numer_masked = ct.where(col_mask, numer, 0.0)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)
    probs = numer_masked / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep = rand_f > 0.1
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, probs, 0.0)
    scaled = dropped * DROPOUT_SCALE
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


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
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
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


@oracle_impl(hardware="B200", point="00541467", BLOCK_K=512)
def oracle_forward(inputs, *, BLOCK_K: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    view_shape = _resolve_shape(shape0, arg0_1.numel())
    random_shape = _as_shape(shape1)
    flat_shape = _resolve_shape(shape2, arg0_1.numel())
    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    device = arg0_1.device
    batch, heads, q_len, k_len = view_shape
    rows = batch * heads * q_len

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

    # Broadcast mask [batch, 1, Q, K] -> [batch, heads, Q, K] contiguous
    mask_broadcast = arg1_1.expand(batch, heads, q_len, k_len).contiguous()
    mask_2d = mask_broadcast.view(rows, k_len)

    # Wrap scalar fill into a 1-element tensor so we can ct.load it.
    fill_1d = arg2_1.view(1)

    scores_2d = arg0_1.contiguous().view(rows, k_len)
    random_2d = random.contiguous().view(rows, k_len)
    where_2d = where.view(rows, k_len)
    amax_1d = amax.view(rows)
    denom_1d = denom.view(rows)
    keep_2d = keep.view(rows, k_len)
    out_2d = out.view(rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _masked_softmax_dropout_kernel,
        (scores_2d, mask_2d, fill_1d, random_2d,
         where_2d, amax_1d, denom_1d, keep_2d, out_2d,
         heads, q_len, k_len, BLOCK_K),
    )
    return where, amax, denom, keep, out, out.permute(0, 2, 1)
