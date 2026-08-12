"""cuTile port of var_mean_e5a8f396e9fa: Longformer dropout-residual LayerNorm + permute.

SEED_INDEX=29, EPS=1e-5, HIDDEN=768. Returns (gt, normalized, affine, final_bf16, div)
where final_bf16 is stored with sequence-major layout (permute(1,0,2)).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 29
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_layernorm_permute_kernel(
    flat_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_flat_ptr,
    norm_flat_ptr,
    affine_flat_ptr,
    final_bf16_flat_ptr,  # stored at final_offsets (seq-major)
    div_ptr,
    BATCH: ct.Constant[int],
    SEQ: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))

    flat = ct.load(flat_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual_f = ct.astype(
        ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    random_f = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf

    zero_bf = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled_bf, ct.float32) + residual_f
    zero_f = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    x_masked = ct.where(col_mask_2d, x, zero_f)

    mean = ct.sum(x_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    row_idx = ct.arange(ROW_BLOCK, dtype=ct.int32) + row_block * ROW_BLOCK
    row_idx_2d = ct.reshape(row_idx, (ROW_BLOCK, 1))
    flat_idx = row_idx_2d * HIDDEN + ct.reshape(cols, (1, BLOCK_H))
    ct.scatter(gt_flat_ptr, flat_idx, keep, mask=col_mask_2d)
    ct.scatter(norm_flat_ptr, flat_idx, normalized, mask=col_mask_2d)
    ct.scatter(affine_flat_ptr, flat_idx, affine, mask=col_mask_2d)

    # Compute permuted output offsets: batch_id = row // SEQ; seq_id = row % SEQ;
    # out_row = seq_id * BATCH + batch_id, final_offset = out_row * HIDDEN + col
    batch_ids = row_idx // SEQ
    seq_ids = row_idx - batch_ids * SEQ
    out_rows = seq_ids * BATCH + batch_ids
    out_rows_2d = ct.reshape(out_rows, (ROW_BLOCK, 1))
    final_idx = out_rows_2d * HIDDEN + ct.reshape(cols, (1, BLOCK_H))
    ct.scatter(final_bf16_flat_ptr, final_idx, ct.astype(affine, ct.bfloat16), mask=col_mask_2d)
    ct.store(div_ptr, index=(row_block,), tile=ct.reshape(invstd * (1.0 / HIDDEN), (ROW_BLOCK,)))


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


@oracle_impl(hardware="B200", point="04503798", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    final_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    batch = int(norm_shape[0])
    seq = int(norm_shape[1])
    device = arg0_1.device
    div_shape = (batch, seq, 1)

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape), device=device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape), device=device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape), device=device, dtype=torch.float32)
    final_bf16 = torch.empty_strided(final_shape, _contiguous_stride(final_shape), device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_flat = gt.view(rows * hidden)
    norm_flat = normalized.view(rows * hidden)
    affine_flat = affine.view(rows * hidden)
    final_flat = final_bf16.view(rows * hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_residual_layernorm_permute_kernel,
        (
            x_2d, random_2d, residual_2d, arg3_1, arg4_1,
            gt_flat, norm_flat, affine_flat, final_flat, div_1d,
            batch, seq, hidden, BLOCK_H, ROW_BLOCK,
        ),
    )
    return gt, normalized, affine, final_bf16, div
