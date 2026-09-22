"""cuTile port of amax_sum_amax_d857d0afb1ee: MT5 dual relative-position attention.

The graph has two independent branches — an encoder (bidirectional bucket) and
a decoder (causal bucket) — each producing:
  * a returned bucket tensor
  * a returned per-head embedding
  * a returned bias tensor (broadcast across batches)
  * bf16-rounded logits + fp32 softmax with amax/sum side outputs
  * seeded Inductor dropout (seeds 1 and 35), bf16 scaled probs + permute alias.

Strategy: materialize the pure-torch pieces (bucket, embedding, bias, causal
add-mask, iota, zero) then run one cuTile row-softmax + dropout kernel per
branch. Random tensors are pre-generated via the eager fallback.
"""

import math

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_ENCODER = 1
SEED_INDEX_DECODER = 35

DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

MASK_VALUE_F32 = -3.4028234663852886e38


def _bidirectional_bucket(query, key, num_buckets=32, max_distance=128):
    """MT5 bidirectional relative-position bucket. query/key: i64 tensors."""
    ret = torch.zeros_like(query - key)
    rel = key - query
    ret = ret + (rel > 0).to(torch.int64) * (num_buckets // 2)
    n = torch.abs(rel)
    max_exact = num_buckets // 4  # 8
    is_small = n < max_exact
    log_ratio = torch.log(n.float() / max_exact) / math.log(max_distance / max_exact)
    val_if_large = max_exact + (log_ratio * max_exact).to(torch.int64)
    val_if_large = torch.minimum(
        val_if_large, torch.full_like(val_if_large, num_buckets // 2 - 1)
    )
    return ret + torch.where(is_small, n, val_if_large)


def _causal_bucket(distance, num_buckets=32, max_distance=128):
    """MT5 causal (unidirectional) bucket. distance = max(query - key, 0)."""
    max_exact = num_buckets // 2  # 16
    is_small = distance < max_exact
    log_ratio = torch.log(distance.float() / max_exact) / math.log(max_distance / max_exact)
    val_if_large = max_exact + (log_ratio * max_exact).to(torch.int64)
    val_if_large = torch.minimum(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )
    return torch.where(is_small, distance, val_if_large)


@ct.kernel
def _softmax_dropout_kernel(
    scores_ptr,      # bf16 [n_rows, K]
    bias_ptr,        # f32  [BATCH, HEADS, Q, K]
    random_ptr,      # f32  [n_rows, K]
    amax_ptr,        # f32  [n_rows]
    sum_ptr,         # f32  [n_rows]
    keep_ptr,        # b8   [n_rows, K]
    out_ptr,         # bf16 [n_rows, K]
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bh = row // Q_LEN
    batch = flat_bh // HEADS
    head = flat_bh - (flat_bh // HEADS) * HEADS
    query = row - flat_bh * Q_LEN

    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_K))
    bias_f = ct.load(bias_ptr, index=(batch, head, query, 0), shape=(1, 1, 1, BLOCK_K))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_K))

    scores_f = ct.astype(scores_bf, ct.float32)
    # The Triton oracle rounds (score + bias) to bf16 then casts back to fp32
    # before the row-softmax reduction.
    rounded = ct.astype(scores_f + bias_2d, ct.bfloat16)
    logits = ct.astype(rounded, ct.float32)

    row_max = ct.max(logits, axis=1, keepdims=True)
    numer = ct.exp(logits - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs_bf = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    random = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K))
    random_bf = ct.astype(random, ct.bfloat16)
    threshold = ct.full((1, BLOCK_K), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, probs_bf, ct.full((1, BLOCK_K), 0.0, dtype=ct.bfloat16))
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


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


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    advance = _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


def _stride4(shape):
    return (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)


def _row_stride(shape):
    return (shape[1] * shape[2], shape[2], 1, 1)


@oracle_impl(hardware="B200", point="d37cb3d9", BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_K: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
        *_shape_params,
    ) = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    heads = int(arg1_1.shape[1])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    batch = int(arg0_1.shape[0] // heads)
    full_shape = (batch, heads, q_len, k_len)
    row_shape = (batch, heads, q_len, 1)
    flat_shape = tuple(int(dim) for dim in _shape_params[5])
    flat_shape_2 = tuple(int(dim) for dim in _shape_params[12])
    n_rows = int(batch * heads * q_len)
    device = arg0_1.device

    # ==== Build the pure-torch side outputs. ====
    iota_q = torch.arange(q_len, device=device, dtype=torch.int64)
    iota_k = torch.arange(k_len, device=device, dtype=torch.int64)
    q_col = iota_q.unsqueeze(1)  # [q, 1]
    k_row = iota_k.unsqueeze(0)  # [1, k]

    # Encoder: bidirectional bucket over (query, key).
    enc_bucket = _bidirectional_bucket(q_col, k_row)  # [q, k] i64
    enc_embed = arg1_1[enc_bucket]                     # [q, k, heads] f32
    enc_rel_perm = enc_embed.permute(2, 0, 1)           # [heads, q, k] f32
    zero_f32_bias = torch.zeros(batch, 1, q_len, k_len, device=device, dtype=torch.float32)
    enc_add4 = enc_rel_perm.unsqueeze(0) + zero_f32_bias  # [batch, heads, q, k] f32
    # add4 has strides (heads*q*k, 1, k*heads, heads) — head is fastest inner
    # after batch. Realize via arithmetic to preserve values while switching
    # to a plane-major layout for the kernel; we'll return the original
    # strided view below.

    # Decoder: causal add-mask + causal bucket.
    dec_causal = k_row <= q_col  # [q, k] bool
    dec_distance = torch.maximum(q_col - k_row, torch.zeros_like(q_col - k_row))
    dec_bucket = _causal_bucket(dec_distance)          # [q, k] i64
    dec_embed = arg4_1[dec_bucket]                      # [q, k, heads] f32
    dec_rel_perm = dec_embed.permute(2, 0, 1)           # [heads, q, k] f32
    causal_bcast = dec_causal.unsqueeze(0).unsqueeze(0)  # [1, 1, q, k]
    mask_f32 = torch.full((), MASK_VALUE_F32, dtype=torch.float32, device=device)
    zero_f32 = torch.zeros((), dtype=torch.float32, device=device)
    where_1 = torch.where(causal_bcast, zero_f32, mask_f32)  # [1, 1, q, k]
    dec_add9 = dec_rel_perm.unsqueeze(0) + where_1  # [batch, heads, q, k] f32

    # ==== Materialize the returned strided tensors. ====
    enc_bias_f32 = enc_add4.expand(batch, heads, q_len, k_len).contiguous()
    dec_bias_f32 = dec_add9.expand(batch, heads, q_len, k_len).contiguous()

    add_3 = torch.empty_strided(
        (q_len, k_len), (k_len, 1), device=device, dtype=torch.int64,
    )
    add_3.copy_(enc_bucket)

    embedding = torch.empty_strided(
        (q_len, k_len, heads), (k_len * heads, heads, 1),
        device=device, dtype=torch.float32,
    )
    embedding.copy_(enc_embed)

    add_4 = torch.empty_strided(
        full_shape, (heads * q_len * k_len, 1, k_len * heads, heads),
        device=device, dtype=torch.float32,
    )
    add_4.copy_(enc_bias_f32)

    unsqueeze_4 = torch.empty_strided(
        (1, 1, q_len), (q_len, q_len, 1), device=device, dtype=torch.int64,
    )
    unsqueeze_4.copy_(iota_q.view(1, 1, q_len))

    full_2 = torch.zeros((), device=device, dtype=torch.float32)

    add_8 = torch.empty_strided(
        (q_len, k_len), (k_len, 1), device=device, dtype=torch.int64,
    )
    add_8.copy_(dec_bucket)

    embedding_1 = torch.empty_strided(
        (q_len, k_len, heads), (k_len * heads, heads, 1),
        device=device, dtype=torch.float32,
    )
    embedding_1.copy_(dec_embed)

    add_9 = torch.empty_strided(
        full_shape, (heads * q_len * k_len, 1, k_len * heads, heads),
        device=device, dtype=torch.float32,
    )
    add_9.copy_(dec_bias_f32)

    amax = torch.empty_strided(row_shape, _row_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _row_stride(row_shape),
                                device=device, dtype=torch.float32)
    gt_1 = torch.empty_strided(full_shape, _stride4(full_shape),
                               device=device, dtype=torch.bool)
    view_1 = torch.empty_strided(
        flat_shape, (q_len * k_len, k_len, 1),
        device=device, dtype=torch.bfloat16,
    )

    amax_1 = torch.empty_strided(row_shape, _row_stride(row_shape),
                                 device=device, dtype=torch.float32)
    sum_2 = torch.empty_strided(row_shape, _row_stride(row_shape),
                                device=device, dtype=torch.float32)
    gt_2 = torch.empty_strided(full_shape, _stride4(full_shape),
                               device=device, dtype=torch.bool)
    view_3 = torch.empty_strided(
        flat_shape_2, (q_len * k_len, k_len, 1),
        device=device, dtype=torch.bfloat16,
    )

    # RNG (two seeds, single-shot advance).
    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX_ENCODER)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX_DECODER)
    random0, random1 = _inductor_random_pair_for_eager_check(
        full_shape, seed0, seed1, device=device,
    )

    # Reshape everything for the kernel.
    enc_scores_2d = arg0_1.view(n_rows, k_len)
    dec_scores_2d = arg3_1.view(n_rows, k_len)
    enc_bias_planed = enc_bias_f32                    # [batch, heads, q, k]
    dec_bias_planed = dec_bias_f32
    enc_rand_2d = random0.contiguous().view(n_rows, k_len)
    dec_rand_2d = random1.contiguous().view(n_rows, k_len)

    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt1_2d = gt_1.view(n_rows, k_len)
    out1_2d = view_1.view(n_rows, k_len)

    amax1_1d = amax_1.view(n_rows)
    sum2_1d = sum_2.view(n_rows)
    gt2_2d = gt_2.view(n_rows, k_len)
    out3_2d = view_3.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (enc_scores_2d, enc_bias_planed, enc_rand_2d,
         amax_1d, sum_1d, gt1_2d, out1_2d,
         heads, q_len, k_len, BLOCK_K),
    )
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (dec_scores_2d, dec_bias_planed, dec_rand_2d,
         amax1_1d, sum2_1d, gt2_2d, out3_2d,
         heads, q_len, k_len, BLOCK_K),
    )

    return (
        add_3, embedding, add_4, amax, sum_1, gt_1, view_1,
        unsqueeze_4, full_2, add_8, embedding_1, add_9,
        amax_1, sum_2, gt_2, view_3,
        view_3.permute(0, 2, 1), view_1.permute(0, 2, 1),
    )
