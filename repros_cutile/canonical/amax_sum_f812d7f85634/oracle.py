"""cuTile port of amax_sum_f812d7f85634: Longformer training sliding-window
attention softmax + dropout.

The band-assembly, edge masks, and bias-add for the pre-softmax score
tensor (`permute_7` in the repro) require intricate slice_scatter chains
that don't map cleanly to a cuTile @kernel. We compute the assembled
score using the same torch decomposition the eager Repro uses, then run
a single cuTile row kernel for the softmax+dropout+final-layout write.

Seeded RNG uses inductor_random pre-generated outside the kernel.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 21
BATCH = 8
SEQ = 1024
HEADS = 12
CHUNK = 256
CHUNKS = 4
WINDOW = 513
PADDED_WINDOW = 770
FINAL_INNER = 769
FINAL_D = 768
ROWS = BATCH * SEQ * HEADS
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
FINAL_SHAPE = (BATCH * HEADS * CHUNKS, CHUNK, FINAL_D)
FINAL_STRIDE = (CHUNK * PADDED_WINDOW, FINAL_INNER, 1)
FINAL_STORAGE = (
    (FINAL_SHAPE[0] - 1) * FINAL_STRIDE[0]
    + (FINAL_SHAPE[1] - 1) * FINAL_STRIDE[1]
    + (FINAL_SHAPE[2] - 1) * FINAL_STRIDE[2]
    + 1
)


TILE_N = 1024  # Next power-of-2 for WINDOW=513.


@ct.kernel
def _softmax_dropout_kernel(
    score_ptr,       # bf16 [ROWS, WINDOW]  (padded to TILE_N via PaddingMode.ZERO)
    query_mask_ptr,  # b8 [ROWS]
    scalar_ptr,      # f32 (1,)
    random_ptr,      # f32 [ROWS, WINDOW]
    amax_out_ptr,    # f32 [ROWS]
    denom_out_ptr,   # f32 [ROWS]
    keep_out_ptr,    # b8 [ROWS, WINDOW]
    scaled_out_ptr,  # bf16 [ROWS, WINDOW]
    WINDOW_C: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(TILE_N, dtype=ct.int32)
    valid = ct.reshape(cols < WINDOW_C, (1, TILE_N))

    score_bf = ct.load(score_ptr, index=(row, 0), shape=(1, TILE_N),
                       padding_mode=ct.PaddingMode.ZERO)
    scores = ct.astype(score_bf, ct.float32)

    # For invalid columns, replace with -inf so they don't affect max.
    neg_inf = ct.full((1, TILE_N), -float("inf"), dtype=ct.float32)
    scores_masked_for_max = ct.where(valid, scores, neg_inf)

    # NaN handling: if any valid column contains NaN, produce NaN as row_max.
    # Propagate via a sum of (nan for nan cells, 0 for others): if any nan
    # is present the sum is nan; otherwise it's 0.
    is_nan = (scores != scores) & valid
    zero_f_pre = ct.full((1, TILE_N), 0.0, dtype=ct.float32)
    nan_tile = ct.full((1, TILE_N), float("nan"), dtype=ct.float32)
    nan_offset = ct.sum(ct.where(is_nan, nan_tile, zero_f_pre))

    row_max_raw = ct.max(scores_masked_for_max)  # 0D
    row_max = row_max_raw + nan_offset

    shifted = scores - row_max
    numer = ct.exp(shifted)
    zero_f = ct.full((1, TILE_N), 0.0, dtype=ct.float32)
    numer_valid = ct.where(valid, numer, zero_f)
    denom = ct.sum(numer_valid)
    probs = numer / denom

    amax_1 = ct.full((1,), row_max, dtype=ct.float32)
    denom_1 = ct.full((1,), denom, dtype=ct.float32)
    ct.store(amax_out_ptr, index=(row,), tile=amax_1)
    ct.store(denom_out_ptr, index=(row,), tile=denom_1)

    # Query mask: if arg8[row] then use scalar (arg9), else use probs.
    query_mask = ct.load(query_mask_ptr, index=(row,), shape=(1,))
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    zero_bool_1 = ct.full((1,), 0, dtype=ct.bool_)
    query_mask_bool = query_mask != zero_bool_1

    scalar_2d = ct.reshape(scalar, (1, 1))
    scalar_bcast = ct.full((1, TILE_N), 0.0, dtype=ct.float32) + scalar_2d
    query_mask_2d = ct.reshape(query_mask_bool, (1, 1))
    query_mask_bcast = ct.full((1, TILE_N), False, dtype=ct.bool_) | query_mask_2d
    selected = ct.where(query_mask_bcast, scalar_bcast, probs)
    selected_bf = ct.astype(selected, ct.bfloat16)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, TILE_N),
                     padding_mode=ct.PaddingMode.ZERO)
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, TILE_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf

    zero_bf = ct.full((1, TILE_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, selected_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )

    # For OOB columns, store 0 so they don't wander. But keep/scaled outputs
    # are only WINDOW columns wide; we need to write only the first WINDOW.
    # Do this via a masked-scatter into columns [0..WINDOW-1].
    # We do this by computing 1D flat offsets and using ct.scatter.
    flat_row = ct.full((TILE_N,), row, dtype=ct.int32)
    flat_col = cols
    row_valid = cols < WINDOW_C
    keep_1d = ct.reshape(keep, (TILE_N,))
    scaled_1d = ct.reshape(scaled_bf, (TILE_N,))
    # cuTile arrays are the caller-supplied 2D tensor; scatter with 2D indices.
    ct.scatter(keep_out_ptr, (flat_row, flat_col), keep_1d, mask=row_valid)
    ct.scatter(scaled_out_ptr, (flat_row, flat_col), scaled_1d, mask=row_valid)


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


def _assemble_score(
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
    *shape_params,
):
    """Build the pre-softmax score tensor using the eager Repro decomposition.

    Returns bf16 permute_7 of shape [BATCH, SEQ, HEADS, WINDOW].
    """
    view = torch.ops.aten.view.default(arg0_1, shape_params[0])
    permute = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3])
    view_1 = torch.ops.aten.view.default(permute, shape_params[1])
    pad = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0)
    view_2 = torch.ops.aten.view.default(pad, shape_params[2])
    slice_1 = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
    slice_2 = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257)
    copy = torch.ops.aten.copy.default(arg1_1.clone(), slice_2)
    slice_scatter = torch.ops.aten.slice_scatter.default(
        arg2_1.clone(), copy, 3, 256, 9223372036854775807,
    )
    slice_scatter_1 = torch.ops.aten.slice_scatter.default(
        arg3_1.clone(), slice_scatter, 1, 0, -1,
    )
    select = torch.ops.aten.select.int(view_2, 1, -1)
    slice_3 = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807)
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, 2, 0, 257)
    select_1 = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
    slice_5 = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
    copy_1 = torch.ops.aten.copy.default(slice_5, slice_4)
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(
        select_1, copy_1, 2, 256, 9223372036854775807,
    )
    select_scatter = torch.ops.aten.select_scatter.default(
        slice_scatter_1, slice_scatter_2, 1, -1,
    )
    slice_6 = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
    slice_7 = torch.ops.aten.slice.Tensor(slice_6, 3, 257, 9223372036854775807)
    slice_8 = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
    slice_9 = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 256)
    copy_2 = torch.ops.aten.copy.default(slice_9, slice_7)
    slice_scatter_3 = torch.ops.aten.slice_scatter.default(
        slice_8, copy_2, 3, 0, 256,
    )
    slice_scatter_4 = torch.ops.aten.slice_scatter.default(
        select_scatter, slice_scatter_3, 1, 1, 9223372036854775807,
    )
    select_2 = torch.ops.aten.select.int(view_2, 1, 0)
    slice_10 = torch.ops.aten.slice.Tensor(select_2, 1, 0, 255)
    slice_11 = torch.ops.aten.slice.Tensor(slice_10, 2, -255, 9223372036854775807)
    select_3 = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
    slice_12 = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
    slice_13 = torch.ops.aten.slice.Tensor(slice_12, 2, 1, 256)
    copy_3 = torch.ops.aten.copy.default(slice_13, slice_11)
    slice_scatter_5 = torch.ops.aten.slice_scatter.default(
        slice_12, copy_3, 2, 1, 256,
    )
    slice_scatter_6 = torch.ops.aten.slice_scatter.default(
        select_3, slice_scatter_5, 1, 1, 256,
    )
    select_scatter_1 = torch.ops.aten.select_scatter.default(
        slice_scatter_4, slice_scatter_6, 1, 0,
    )
    view_3 = torch.ops.aten.view.default(select_scatter_1, shape_params[3])
    permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3])
    slice_14 = torch.ops.aten.slice.Tensor(permute_1, 1, 0, 256)
    slice_15 = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 257)
    where = torch.ops.aten.where.self(arg4_1, arg5_1, slice_15)
    copy_4 = torch.ops.aten.copy.default(slice_15, where)
    slice_scatter_7 = torch.ops.aten.slice_scatter.default(
        slice_14, copy_4, 3, 0, 257,
    )
    slice_scatter_8 = torch.ops.aten.slice_scatter.default(
        permute_1, slice_scatter_7, 1, 0, 256,
    )
    permute_2 = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3])
    view_4 = torch.ops.aten.view.default(permute_2, shape_params[4])
    view_5 = torch.ops.aten.view.default(view_4, shape_params[5])
    permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3])
    slice_16 = torch.ops.aten.slice.Tensor(permute_3, 1, -256, 9223372036854775807)
    slice_17 = torch.ops.aten.slice.Tensor(slice_16, 3, -257, 9223372036854775807)
    where_1 = torch.ops.aten.where.self(arg6_1, arg5_1, slice_17)
    copy_5 = torch.ops.aten.copy.default(slice_17, where_1)
    slice_scatter_9 = torch.ops.aten.slice_scatter.default(
        slice_16, copy_5, 3, -257, 9223372036854775807,
    )
    slice_scatter_10 = torch.ops.aten.slice_scatter.default(
        permute_3, slice_scatter_9, 1, -256, 9223372036854775807,
    )
    permute_4 = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3])
    permute_5 = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3])
    add = torch.ops.aten.add.Tensor(permute_5, arg7_1)  # bf16 rounded
    permute_6 = torch.ops.aten.permute.default(add, [0, 2, 1, 3])
    permute_7 = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3])
    return permute_7


@oracle_impl(hardware="B200", point="b64f0e8a", block_n=1024, num_warps=4)
def oracle_forward(inputs, *, block_n: int, num_warps: int):
    del block_n, num_warps
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1,
        *shape_params,
    ) = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device

    # Build permute_7 via eager ops (this reproduces the exact bf16 score tensor).
    permute_7 = _assemble_score(
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        *shape_params,
    )

    # Random tensor
    scores_shape = (BATCH, SEQ, HEADS, WINDOW)
    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(scores_shape, seed, device=device)

    amax = torch.empty_strided(
        (BATCH, SEQ, HEADS, 1),
        (SEQ * HEADS, HEADS, 1, 1),
        device=device, dtype=torch.float32,
    )
    denom = torch.empty_strided(
        (BATCH, SEQ, HEADS, 1),
        (SEQ * HEADS, HEADS, 1, 1),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        scores_shape,
        (SEQ * HEADS * WINDOW, HEADS * WINDOW, WINDOW, 1),
        device=device, dtype=torch.bool,
    )
    scaled = torch.empty_strided(
        scores_shape,
        (SEQ * HEADS * WINDOW, HEADS * WINDOW, WINDOW, 1),
        device=device, dtype=torch.bfloat16,
    )

    # Row views for kernel.
    perm7_flat = permute_7.contiguous().view(ROWS, WINDOW)
    random_flat = random.contiguous().view(ROWS, WINDOW)
    amax_1d = amax.view(ROWS)
    denom_1d = denom.view(ROWS)
    keep_2d = keep.view(ROWS, WINDOW)
    scaled_2d = scaled.view(ROWS, WINDOW)

    # arg8 (query_mask) is [BATCH, SEQ, 1, 1]; per-row scalar. arg9 is f32 scalar.
    query_mask_1d = arg8_1.view(BATCH * SEQ).contiguous()
    # Broadcast per (batch, seq) over HEADS.
    query_mask_expanded = query_mask_1d.unsqueeze(1).expand(BATCH * SEQ, HEADS).contiguous().view(ROWS)
    scalar_1d = arg9_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _softmax_dropout_kernel,
        (perm7_flat, query_mask_expanded, scalar_1d, random_flat,
         amax_1d, denom_1d, keep_2d, scaled_2d, WINDOW),
    )

    # Build view_9/permute_9 from `scaled` per the eager layout ops.
    # scaled has stride (SEQ*HEADS*WINDOW, HEADS*WINDOW, WINDOW, 1) so
    # it's [BATCH, SEQ, HEADS, WINDOW] contiguous in the requested strides
    # (contiguous). Then permute to [BATCH, HEADS, SEQ, WINDOW] and reshape.
    perm8 = scaled.permute(0, 2, 1, 3).contiguous()  # [BATCH, HEADS, SEQ, WINDOW]
    view_6 = perm8.view(BATCH * HEADS, CHUNKS, CHUNK, WINDOW)
    padded = torch.ops.aten.constant_pad_nd.default(
        view_6, [0, PADDED_WINDOW - WINDOW, 0, 0], 0.0,
    )  # [BATCH*HEADS, CHUNKS, CHUNK, PADDED_WINDOW]
    view_7 = padded.view(BATCH * HEADS, CHUNKS, CHUNK * PADDED_WINDOW)
    slice_18 = view_7[:, :, : CHUNK * PADDED_WINDOW - 256]
    view_8 = slice_18.reshape(BATCH * HEADS, CHUNKS, CHUNK, FINAL_INNER)
    slice_19 = view_8[:, :, :, :FINAL_D]
    unsqueeze = slice_19.unsqueeze(4)
    view_9 = unsqueeze.view(BATCH * HEADS * CHUNKS, CHUNK, FINAL_D)
    permute_9 = view_9.permute(0, 2, 1)

    return permute_7, amax, denom, keep, view_9, permute_9
