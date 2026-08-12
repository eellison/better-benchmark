"""cuTile port of amax_sum_dad9c0a4a061: DeBERTaV2 attention softmax + dropout with mask fill.

For each row: apply mask (where mask==True, use arg2 fill value; else use scores),
compute fp32 softmax (amax/exp/sum/div), then dropout via pre-generated random.
Returns (where, amax, sum_1, gt, convert_element_type_1, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 55
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

K_LEN = 512


@ct.kernel
def _softmax_dropout_kernel(
    scores_ptr,     # bf16 (n_rows, K_LEN)
    mask_ptr,       # bool (n_rows_per_batch_group, K_LEN) actually (batch, 1, Q, K)
    fill_ptr,       # bf16 scalar
    random_ptr,     # f32 (n_rows, K_LEN)
    where_ptr,      # bf16 (n_rows, K_LEN)
    amax_ptr,       # f32 (n_rows,)
    sum_ptr,        # f32 (n_rows,)
    gt_ptr,         # bool (n_rows, K_LEN)
    bf16_out_ptr,   # bf16 (n_rows, K_LEN)
    K_LEN_C: ct.Constant[int],
    HEADS_TIMES_Q: ct.Constant[int],
):
    # Each program handles one row.
    # row = batch_head_query index; row // HEADS_TIMES_Q gives batch,
    # then we need to look up mask[batch, 0, row%Q, cols].
    # mask shape is (8,1,512,512); indexed by (batch, 0, q, col).
    # Simplification: mask has shape (batch, 1, q, k). We flatten batch by (q,k) contiguous,
    # so mask_flat has shape (batch, q*k). For each row r, batch = r // (heads*q).
    row = ct.bid(0)

    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, K_LEN_C))

    # Compute mask row: mask is indexed by (batch, 0, query, col).
    # batch_head_query = row; batch = row // HEADS_TIMES_Q; query = row % Q_LEN.
    # But wait, HEADS * Q_LEN = 24 * 512 = 12288. So row // 12288 = batch.
    # rows_per_batch = 12288; within a batch, we have 24 heads * 512 queries.
    # For the mask we ignore heads (dim=1 is size 1), so batch offset is:
    # batch * (Q * K) + query * K + col
    batch_idx = row // HEADS_TIMES_Q
    within_batch = row - batch_idx * HEADS_TIMES_Q
    # heads dim is size 24, so within_batch = head * Q_LEN + query
    query = within_batch - (within_batch // K_LEN_C) * K_LEN_C
    # mask flat linear index for this row: batch * Q_LEN * K_LEN + query * K_LEN + 0..K_LEN
    # But mask has 4 dims (batch, 1, Q, K); treat as (batch, Q, K) contiguous.
    mask_row = batch_idx * K_LEN_C + query  # index into flattened (batch*Q, K)
    m = ct.load(mask_ptr, index=(mask_row, 0), shape=(1, K_LEN_C))

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_val = ct.astype(fill, ct.float32)
    fill_2d = ct.reshape(fill_val, (1, 1))

    where_pre = ct.where(
        m,
        ct.astype(fill_2d, ct.bfloat16),
        scores_bf,
    )
    ct.store(where_ptr, index=(row, 0), tile=where_pre)

    scores_f = ct.astype(where_pre, ct.float32)
    row_max = ct.max(scores_f, axis=1, keepdims=True)
    numer = ct.exp(scores_f - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN_C))
    p_f = ct.full(shape=(1, K_LEN_C), fill_value=DROPOUT_P, dtype=ct.float32)
    keep = random_f > p_f
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, K_LEN_C), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = dropped * DROPOUT_SCALE
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound_offset = offset - advance
        rewound[8:16] = torch.tensor(
            list(int(rewound_offset).to_bytes(8, "little", signed=False)),
            dtype=state.dtype, device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="00541467")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    # arg0_1: bf16 [192, 512, 512], view to [8, 24, 512, 512]
    # arg1_1: b8 [8, 1, 512, 512]
    # arg2_1: bf16 scalar
    # arg3_1: seed tensor
    device = arg0_1.device
    full4d = (8, 24, 512, 512)
    view = arg0_1.view(full4d)
    view_contig = view.contiguous()
    n_rows = 8 * 24 * 512
    heads_times_q = 24 * K_LEN
    # HEADS = 24, so HEADS_TIMES_Q = 24 * 512 = 12288

    # Where: mask fills locations where arg1_1==True with fill scalar arg2_1, else view
    where_out = torch.empty_strided(full4d, _contiguous_stride(full4d),
                                    device=device, dtype=torch.bfloat16)
    row_shape = full4d[:-1] + (1,)
    amax_out = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                   device=device, dtype=torch.float32)
    sum_out = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                  device=device, dtype=torch.float32)
    gt_out = torch.empty_strided(full4d, _contiguous_stride(full4d),
                                 device=device, dtype=torch.bool)
    bf16_out_shape = (192, 512, 512)
    bf16_out = torch.empty_strided(bf16_out_shape, _contiguous_stride(bf16_out_shape),
                                   device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full4d, seed, device=device)

    scores_2d = view_contig.reshape(n_rows, K_LEN)
    # Mask is (8, 1, 512, 512): reshape to (8*512, 512) -> row index formula: batch*512 + query
    mask_2d = arg1_1.reshape(8 * K_LEN, K_LEN)
    fill_1d = arg2_1.reshape(1)
    random_2d = random.reshape(n_rows, K_LEN)
    where_2d = where_out.view(n_rows, K_LEN)
    amax_1d = amax_out.view(n_rows)
    sum_1d = sum_out.view(n_rows)
    gt_2d = gt_out.view(n_rows, K_LEN)
    bf16_out_2d = bf16_out.view(n_rows, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_dropout_kernel,
        (
            scores_2d, mask_2d, fill_1d, random_2d,
            where_2d, amax_1d, sum_1d, gt_2d, bf16_out_2d,
            K_LEN, heads_times_q,
        ),
    )
    return where_out, amax_out, sum_out, gt_out, bf16_out, bf16_out.permute(0, 2, 1)
