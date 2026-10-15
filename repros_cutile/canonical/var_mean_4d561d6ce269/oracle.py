"""cuTile port of var_mean_4d561d6ce269: RoBERTa embedding LayerNorm + dropout.

Pre-computes the position/token-type ids and embedding lookups in torch, then
uses cuTile to do the LayerNorm row-reduction, affine epilogue, dropout mask,
and scaled output. HIDDEN=768 with BLOCK_H=1024 uses padded loads/stores.

Note the outputs `inductor_seeds`, `gt`, `mul_4`, and `view` are all stochastic
(they depend on the RNG state at forward time). The `add_1`, `expand_1`,
`mul_1`, and `div` outputs are deterministic and must match eager exactly.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
SEED_COUNT = 37
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
BLOCK_H = 1024


@ct.kernel
def _ln_dropout_kernel(
    embedded_ptr,   # f32 [rows, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    random_ptr,     # f32 [rows, HIDDEN]
    normalized_pad_ptr,   # f32 [rows, BLOCK_H]
    keep_pad_ptr,   # bool [rows, BLOCK_H]
    scaled_pad_ptr, # f32 [rows, BLOCK_H]
    bf16_view_pad_ptr,    # bf16 [rows, BLOCK_H]
    div_ptr,        # f32 [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        embedded_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, 0.0)
    row_sum = ct.sum(x_masked)
    mean = row_sum * (1.0 / HIDDEN_)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d

    random = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep = random > DROPOUT_P
    dropped = ct.astype(keep, ct.float32) * affine
    scaled = dropped * DROPOUT_SCALE

    ct.store(normalized_pad_ptr, index=(row, 0), tile=normalized)
    ct.store(keep_pad_ptr, index=(row, 0), tile=keep)
    ct.store(scaled_pad_ptr, index=(row, 0), tile=scaled)
    ct.store(bf16_view_pad_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN_), (1,)))


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


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


@oracle_impl(hardware="B200", point="7fadcbae", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, **_kwargs):
    (
        cumsum,
        position_mask,
        token_type_source,
        word_table,
        word_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        expand_shape0,
        expand_shape1,
        random_shape,
        flat_shape,
    ) = inputs
    random_shape = _as_shape(random_shape)
    flat_shape = _as_shape(flat_shape)
    device = word_table.device

    # Build position ids and token type ids in torch (matching the eager Repro).
    # position_id = (cumsum.to(int32) * position_mask).to(int64) per row
    cumsum_i32 = cumsum.to(torch.int32)
    position_id = (cumsum_i32 * position_mask).to(torch.int64)  # [BATCH, SEQ]
    # position_ids returned as [BATCH, SEQ] i64
    position_ids_out = position_id.contiguous()

    # token_type_source shape [1, 512]; gather per (batch, seq) using position_id
    # The Repro does: expand(arg2_1, [BATCH, SEQ]) then gather along dim=1 by
    # position_id. Since token_type_source is [1, SEQ] and we expand to [BATCH, SEQ],
    # each batch has the same [SEQ] row. Gathering along dim=1 by position_id
    # index j gives token_type_source[0, position_id_row_val] for each element.
    expanded_source = token_type_source.expand(BATCH, SEQ_LEN)  # [BATCH, SEQ]
    token_type_ids = torch.gather(expanded_source, 1, position_id)  # [BATCH, SEQ]

    # expand_1 is expand of gather to [BATCH, SEQ] — same as token_type_ids
    expand_1 = token_type_ids.contiguous()

    # Embedding lookups
    word_embed = word_table[word_ids]                            # [BATCH, SEQ, 768]
    token_type_embed = token_type_table[token_type_ids]          # [BATCH, SEQ, 768]
    position_embed = position_table[position_id]                 # [BATCH, SEQ, 768]
    embedded = word_embed + token_type_embed + position_embed    # [BATCH, SEQ, 768]

    # Pre-generate the seeds and random tensor for dropout.
    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    # cuTile kernel over rows.
    embedded_flat = embedded.reshape(ROWS, HIDDEN).contiguous()
    random_flat = random.reshape(ROWS, HIDDEN).contiguous()

    normalized_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    keep_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bool)
    scaled_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    bf16_view_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _ln_dropout_kernel,
        (embedded_flat, weight, bias, random_flat,
         normalized_pad, keep_pad, scaled_pad, bf16_view_pad, div_1d,
         HIDDEN, BLOCK_H),
    )

    # Materialize output tensors matching eager strides.
    normalized = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.float32,
    )
    normalized.view(ROWS, HIDDEN).copy_(normalized_pad.narrow(1, 0, HIDDEN))

    keep = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    keep.view(ROWS, HIDDEN).copy_(keep_pad.narrow(1, 0, HIDDEN))

    scaled = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.float32,
    )
    scaled.view(ROWS, HIDDEN).copy_(scaled_pad.narrow(1, 0, HIDDEN))

    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    bf16_view.view(ROWS, HIDDEN).copy_(bf16_view_pad.narrow(1, 0, HIDDEN))

    div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1), (SEQ_LEN, 1, 1),
        device=device, dtype=torch.float32,
    )
    div.view(ROWS).copy_(div_1d)

    # Match eager return order: (add_1, expand_1, mul_1, inductor_seeds, gt, mul_4, view, div)
    # add_1 = position_id, expand_1 = token_type_ids, mul_1 = normalized (f32),
    # inductor_seeds = seeds, gt = keep, mul_4 = scaled, view = bf16_view, div = div
    return position_ids_out, expand_1, normalized, seeds, keep, scaled, bf16_view, div
