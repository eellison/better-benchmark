"""cuTile port of amax_sum_884c406c2df8: Longformer sliding-window attention.

Strategy:
  * torch: build the assembled score tensor `permute_7` by replaying the
    Repro's slice_scatter / select_scatter / view+permute band-assembly.
    This is torch's strong suit; there's little to gain porting it to cuTile.
  * cuTile: single row kernel over 8*1024*12 = 98304 rows of length 513 that
    does stable softmax + apply query_mask + seeded dropout + scaled output.
    Produces amax, sum_1, and dropout mask + scaled bf16 output.
  * torch: pad+slice+reshape the scaled dropout output into the final
    padded layout (`view_9`, `permute_9`).

Non-graph-capture path only (seeded RNG must be materialized ahead of the
kernel via inductor_random). If the caller runs under `torch.cuda.graph()`,
we raise NotImplementedError to signal the stub keeps.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

BATCH = 8
SEQ = 1024
HEADS = 12
WINDOW = 513
FINAL_INNER = 769
FINAL_D = 768
PADDED_WINDOW = 770
CHUNK = 256
CHUNKS = 4
GROUPS = BATCH * HEADS  # 96
ROWS = BATCH * SEQ * HEADS  # 98304


@ct.kernel
def _longformer_softmax_dropout_kernel(
    scores_ptr,      # bf16 [ROWS, WINDOW]
    query_mask_ptr,  # b8   [ROWS]  broadcast of query_mask[b, seq]
    scalar_ptr,      # f32 scalar (viewed as [1])
    random_ptr,      # f32  [ROWS, WINDOW]  pre-generated inductor_random
    amax_ptr,        # f32  [ROWS]
    denom_ptr,       # f32  [ROWS]
    keep_ptr,        # b8   [ROWS, WINDOW]
    dropped_ptr,     # bf16 [ROWS, WINDOW]  scaled dropout output
    WINDOW_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    valid = ct.reshape(cols < WINDOW_, (1, BLOCK_N))

    # Load bf16 scores; ZERO padding leaves OOB columns as 0 which we'll
    # replace with -inf below.
    scores_bf = ct.load(
        scores_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scores_f = ct.astype(scores_bf, ct.float32)
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    scores_masked = ct.where(valid, scores_f, neg_inf)

    # Detect NaN: if any valid col is NaN, whole row's max becomes NaN.
    is_nan = scores_masked != scores_masked
    has_nan_i = ct.max(ct.astype(is_nan, ct.int32), axis=1)  # (1,)
    row_max_raw = ct.max(scores_masked, axis=1)  # (1,)
    nan_val = ct.full((1,), float("nan"), dtype=ct.float32)
    row_max = ct.where(has_nan_i != 0, nan_val, row_max_raw)
    row_max_2d = ct.reshape(row_max, (1, 1))

    numer = ct.exp(scores_masked - row_max_2d)
    zero_f = ct.full((1, BLOCK_N), 0.0, dtype=ct.float32)
    numer_masked = ct.where(valid, numer, zero_f)
    denom = ct.sum(numer_masked, axis=1)  # (1,)
    denom_2d = ct.reshape(denom, (1, 1))
    probs = numer_masked / denom_2d  # (1, BLOCK_N) f32 — matches div output

    ct.store(amax_ptr, index=(row,), tile=row_max)
    ct.store(denom_ptr, index=(row,), tile=denom)

    # Apply query_mask: where(query_mask, scalar, probs) then cast to bf16.
    qmask = ct.load(query_mask_ptr, index=(row,), shape=(1,))
    qmask_2d = ct.reshape(qmask, (1, 1))
    scalar_tile = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_2d = ct.reshape(scalar_tile, (1, 1))
    selected = ct.where(qmask_2d, scalar_2d, probs)
    selected_bf = ct.astype(selected, ct.bfloat16)

    # Seeded dropout: keep = random > 0.1  (both rounded to bf16).
    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32), ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, selected_bf, zero_bf)
    # bf16 mul via f32 round: mul = dropped * 1.1111... rounded to bf16
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_bf)


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


def _assemble_scores(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
                     arg7_1, shape_params):
    """Replay the Repro's band-assembly to produce permute_7 bf16 [8,1024,12,513]."""
    (
        _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3,
        _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7,
        _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11,
    ) = shape_params

    view = torch.ops.aten.view.default(arg0_1, _shape_param_0)
    permute = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3])
    view_1 = torch.ops.aten.view.default(permute, _shape_param_1)
    constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
        view_1, [0, 0, 0, 1], 0.0)
    view_2 = torch.ops.aten.view.default(constant_pad_nd, _shape_param_2)
    slice_1 = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
    slice_2 = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257)
    copy = torch.ops.aten.copy.default(arg1_1, slice_2)
    slice_scatter = torch.ops.aten.slice_scatter.default(
        arg2_1, copy, 3, 256, 9223372036854775807)
    slice_scatter_1 = torch.ops.aten.slice_scatter.default(
        arg3_1, slice_scatter, 1, 0, -1)
    select = torch.ops.aten.select.int(view_2, 1, -1)
    slice_3 = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807)
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, 2, 0, 257)
    select_1 = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
    slice_5 = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
    copy_1 = torch.ops.aten.copy.default(slice_5, slice_4)
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(
        select_1, copy_1, 2, 256, 9223372036854775807)
    select_scatter = torch.ops.aten.select_scatter.default(
        slice_scatter_1, slice_scatter_2, 1, -1)
    slice_6 = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
    slice_7 = torch.ops.aten.slice.Tensor(slice_6, 3, 257, 9223372036854775807)
    slice_8 = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
    slice_9 = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 256)
    copy_2 = torch.ops.aten.copy.default(slice_9, slice_7)
    slice_scatter_3 = torch.ops.aten.slice_scatter.default(
        slice_8, copy_2, 3, 0, 256)
    slice_scatter_4 = torch.ops.aten.slice_scatter.default(
        select_scatter, slice_scatter_3, 1, 1, 9223372036854775807)
    select_2 = torch.ops.aten.select.int(view_2, 1, 0)
    slice_10 = torch.ops.aten.slice.Tensor(select_2, 1, 0, 255)
    slice_11 = torch.ops.aten.slice.Tensor(slice_10, 2, -255, 9223372036854775807)
    select_3 = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
    slice_12 = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
    slice_13 = torch.ops.aten.slice.Tensor(slice_12, 2, 1, 256)
    copy_3 = torch.ops.aten.copy.default(slice_13, slice_11)
    slice_scatter_5 = torch.ops.aten.slice_scatter.default(
        slice_12, copy_3, 2, 1, 256)
    slice_scatter_6 = torch.ops.aten.slice_scatter.default(
        select_3, slice_scatter_5, 1, 1, 256)
    select_scatter_1 = torch.ops.aten.select_scatter.default(
        slice_scatter_4, slice_scatter_6, 1, 0)
    view_3 = torch.ops.aten.view.default(select_scatter_1, _shape_param_3)
    permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3])

    # Apply arg4_1 begin mask on the first-chunk slice.
    slice_14 = torch.ops.aten.slice.Tensor(permute_1, 1, 0, 256)
    slice_15 = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 257)
    where = torch.ops.aten.where.self(arg4_1, arg5_1, slice_15)
    copy_4 = torch.ops.aten.copy.default(slice_15, where)
    slice_scatter_7 = torch.ops.aten.slice_scatter.default(
        slice_14, copy_4, 3, 0, 257)
    slice_scatter_8 = torch.ops.aten.slice_scatter.default(
        permute_1, slice_scatter_7, 1, 0, 256)
    permute_2 = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3])
    view_4 = torch.ops.aten.view.default(permute_2, _shape_param_4)
    view_5 = torch.ops.aten.view.default(view_4, _shape_param_5)
    permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3])

    # Apply arg6_1 end mask on the last-chunk slice.
    slice_16 = torch.ops.aten.slice.Tensor(permute_3, 1, -256, 9223372036854775807)
    slice_17 = torch.ops.aten.slice.Tensor(slice_16, 3, -257, 9223372036854775807)
    where_1 = torch.ops.aten.where.self(arg6_1, arg5_1, slice_17)
    copy_5 = torch.ops.aten.copy.default(slice_17, where_1)
    slice_scatter_9 = torch.ops.aten.slice_scatter.default(
        slice_16, copy_5, 3, -257, 9223372036854775807)
    slice_scatter_10 = torch.ops.aten.slice_scatter.default(
        permute_3, slice_scatter_9, 1, -256, 9223372036854775807)
    permute_4 = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3])
    permute_5 = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3])

    # Add position bias arg7_1 (broadcast across heads).
    add = torch.ops.aten.add.Tensor(permute_5, arg7_1)  # bf16 [8, 1024, 12, 513]
    permute_6 = torch.ops.aten.permute.default(add, [0, 2, 1, 3])
    permute_7 = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3])
    return permute_7  # bf16 [8, 1024, 12, 513]


def _final_layout(scaled_2d, device):
    """Replay the Repro's post-dropout view/pad/reshape to produce view_9."""
    scaled_bf = scaled_2d.view(BATCH, SEQ, HEADS, WINDOW)
    permute_8 = torch.ops.aten.permute.default(scaled_bf, [0, 2, 1, 3])
    clone_1 = torch.ops.aten.clone.default(permute_8, memory_format=torch.contiguous_format)
    view_6 = torch.ops.aten.view.default(clone_1, [96, 4, 256, 513])
    constant_pad_nd_1 = torch.ops.aten.constant_pad_nd.default(view_6, [0, 257], 0.0)
    view_7 = torch.ops.aten.view.default(constant_pad_nd_1, [96, 4, 197120])
    slice_18 = torch.ops.aten.slice.Tensor(view_7, 2, 0, -256)
    view_8 = torch.ops.aten.view.default(slice_18, [96, 4, 256, 769])
    slice_19 = torch.ops.aten.slice.Tensor(view_8, 3, 0, -1)
    unsqueeze = torch.ops.aten.unsqueeze.default(slice_19, 4)
    view_9 = torch.ops.aten.view.default(unsqueeze, [384, 256, 768])
    permute_9 = torch.ops.aten.permute.default(view_9, [0, 2, 1])
    return view_9, permute_9


@oracle_impl(hardware="B200", point="b64f0e8a", block_n=1024)
def oracle_forward(inputs, *, block_n: int):
    (
        arg0_1,   # bf16 [288, 512, 512]  (bmm)
        arg1_1,   # bf16 [96, 3, 256, 257]
        arg2_1,   # bf16 [96, 3, 256, 513]
        arg3_1,   # bf16 [96, 4, 256, 513]  <- base tensor (aliased target)
        arg4_1,   # b8   [8, 256, 12, 257] begin-mask
        arg5_1,   # bf16 [8, 256, 12, 257] edge-fill values
        arg6_1,   # b8   [8, 256, 12, 257] end-mask
        arg7_1,   # bf16 [8, 1024, 1, 513] position bias
        arg8_1,   # b8   [8, 1024, 1, 1]  query-mask
        arg9_1,   # f32  scalar
        arg10_1,  # i64  [36]  seeds
        *shape_params,
    ) = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    # Assemble the score tensor via torch (heavy slice_scatter dance).
    permute_7 = _assemble_scores(
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        shape_params,
    )  # bf16 [8, 1024, 12, 513]

    # Prepare scores as (ROWS, WINDOW) contiguous.
    scores_bf_contig = permute_7.contiguous()
    scores_2d = scores_bf_contig.view(ROWS, WINDOW)

    # Broadcast query_mask [8, 1024, 1, 1] -> [ROWS] (12 heads share it).
    query_mask_expanded = arg8_1.expand(BATCH, SEQ, HEADS, 1).contiguous()
    query_mask_1d = query_mask_expanded.view(ROWS)

    # Scalar tile.
    scalar_1d = arg9_1.view(1)

    # Materialize the seeded random tensor (matches the eager-check path).
    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        (BATCH, SEQ, HEADS, WINDOW), seed, device=device,
    )
    random_2d = random.reshape(ROWS, WINDOW).contiguous()

    # Kernel outputs.
    amax_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    denom_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    keep_2d = torch.empty((ROWS, WINDOW), device=device, dtype=torch.bool)
    dropped_2d = torch.empty((ROWS, WINDOW), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _longformer_softmax_dropout_kernel,
        (scores_2d, query_mask_1d, scalar_1d, random_2d,
         amax_1d, denom_1d, keep_2d, dropped_2d,
         WINDOW, block_n),
    )

    # Reshape outputs into the return signature layouts.
    amax = amax_1d.view(BATCH, SEQ, HEADS, 1)
    sum_1 = denom_1d.view(BATCH, SEQ, HEADS, 1)
    keep = keep_2d.view(BATCH, SEQ, HEADS, WINDOW)
    view_9, permute_9 = _final_layout(dropped_2d, device)
    return permute_7, amax, sum_1, keep, view_9, permute_9
