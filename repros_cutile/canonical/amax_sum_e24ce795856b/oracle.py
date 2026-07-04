"""cuTile port of amax_sum_e24ce795856b: Longformer sliding-window attention softmax+dropout.

The substantive @ct.kernel implements the per-row masked softmax + dropout
epilogue:
    clone = permute_7.to(f32)
    amax  = max(clone, axis=-1)
    exp   = exp(clone - amax)
    sum_1 = sum(exp, axis=-1)
    div   = exp / sum_1
    where = query_mask ? scalar_fill : div       # arg8 broadcast, arg9 scalar
    bf16  = cast(where)
    keep  = random > 0.1                          # from pre-generated inductor_random
    mul_1 = keep * bf16 * 1.1111...

The complex diagonal-band layout that builds `permute_7` (the input to the
softmax) and the padded/scattered `view_9` layout at the tail — all pure
aten slice_scatter/select_scatter/permute/view — is replayed via torch on
the host side, so it bit-matches the eager Repro's intermediate tensors.

Seeded RNG (`inductor_random`) is generated on the host via the standard
Inductor recipe: rewind CUDA stream offset before invocation so the
random tensor matches the one the eager module would produce.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HEADS = 12
WINDOW = 513
ROWS = BATCH * SEQ * HEADS  # 8 * 1024 * 12 = 98304

BLOCK_N = 1024              # power-of-2 tile >= WINDOW
SEED_INDEX = 15
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    perm7_ptr,       # bf16 [ROWS, BLOCK_N] — permute_7 padded to BLOCK_N with zeros
    query_mask_ptr,  # bool [ROWS]           — arg8 broadcast to per-row scalar
    fill_ptr,        # f32  [1]              — arg9 (scalar fill value)
    random_ptr,      # f32  [ROWS, BLOCK_N] — pre-generated inductor_random padded
    amax_ptr,        # f32  [ROWS]           — output: per-row amax
    sum_ptr,         # f32  [ROWS]           — output: per-row sum (softmax denom)
    keep_ptr,        # bool [ROWS, BLOCK_N] — output: dropout mask (padded)
    mul_out_ptr,     # bf16 [ROWS, BLOCK_N] — output: keep * bf16(where) * scale
    valid_mask_ptr,  # bool [BLOCK_N]        — True for cols < WINDOW
    BLOCK: ct.Constant[int],
):
    row = ct.bid(0)

    perm7 = ct.load(perm7_ptr, index=(row, 0), shape=(1, BLOCK))
    perm7_f = ct.astype(perm7, ct.float32)
    valid_mask = ct.load(valid_mask_ptr, index=(0,), shape=(BLOCK,))
    valid_mask_2d = ct.reshape(valid_mask, (1, BLOCK))

    # For amax: mask OOB cols with -inf so they don't contaminate the max.
    neg_inf = ct.full((1, BLOCK), float("-inf"), dtype=ct.float32)
    clone_for_max = ct.where(valid_mask_2d, perm7_f, neg_inf)
    row_max = ct.max(clone_for_max, axis=1, keepdims=True)
    # cuTile ct.max does not propagate NaN (unlike torch.amax); explicitly
    # detect NaN in the valid portion of the row and force NaN into the amax.
    nan_flag = ct.isnan(clone_for_max)
    nan_count = ct.sum(ct.astype(nan_flag, ct.int32), axis=1, keepdims=True)
    has_nan = nan_count > 0
    nan_val = ct.full((1, 1), float("nan"), dtype=ct.float32)
    row_max = ct.where(has_nan, nan_val, row_max)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))

    shifted = perm7_f - row_max
    exp_v = ct.exp(shifted)
    # For sum: zero out OOB cols so they don't contribute.
    zero_f = ct.zeros((1, BLOCK), dtype=ct.float32)
    exp_masked = ct.where(valid_mask_2d, exp_v, zero_f)
    row_sum = ct.sum(exp_masked, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(row_sum, (1,)))

    div_v = exp_v / row_sum

    # Apply query mask (broadcast per-row) and scalar fill.
    q_mask = ct.load(query_mask_ptr, index=(row,), shape=(1,))
    q_mask_2d = ct.reshape(q_mask, (1, 1))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_2d = ct.reshape(fill, (1, 1))
    where_v = ct.where(q_mask_2d, fill_2d, div_v)
    bf16_v = ct.astype(where_v, ct.bfloat16)

    # Dropout: convert random -> bf16, keep = > 0.1.
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK), DROPOUT_P, dtype=ct.bfloat16)
    keep_v = rand_bf > threshold_bf

    # Zero out OOB before storing keep (store writes full tile).
    zero_bool = ct.full((1, BLOCK), False, dtype=ct.bool_)
    keep_masked = ct.where(valid_mask_2d, keep_v, zero_bool)
    ct.store(keep_ptr, index=(row, 0), tile=keep_masked)

    # mul = keep * bf16; mul_1 = mul * 1.1111
    keep_bf = ct.astype(keep_v, ct.bfloat16)
    mul_bf = keep_bf * bf16_v
    scale_bf = ct.full((1, BLOCK), DROPOUT_SCALE, dtype=ct.bfloat16)
    mul_1_bf = mul_bf * scale_bf
    zero_bf = ct.full((1, BLOCK), 0.0, dtype=ct.bfloat16)
    mul_1_masked = ct.where(valid_mask_2d, mul_1_bf, zero_bf)
    ct.store(mul_out_ptr, index=(row, 0), tile=mul_1_masked)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    """Rewind the CUDA RNG offset so `inductor_random` produces the tensor
    the eager module would see. Uses the curand4-per-block accounting so the
    advance value is a multiple of 4 (matching the underlying philox stream)."""
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


def _build_permute_7(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, device):
    """Replay the eager diagonal-band scatter/gather layout that produces
    `permute_7` (bf16[8, 1024, 12, 513]). Pure torch aten ops throughout."""
    view = torch.ops.aten.view.default(arg0_1, [96, 3, 512, 1, 512])
    permute = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3])
    view_1 = torch.ops.aten.view.default(permute, [96, 3, 512, 512])
    pad = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0)
    view_2 = torch.ops.aten.view.default(pad, [96, 3, 512, 513])

    slice_1 = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
    slice_2 = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257)
    copy = torch.ops.aten.copy.default(arg1_1, slice_2)
    ss = torch.ops.aten.slice_scatter.default(arg2_1, copy, 3, 256, 9223372036854775807)
    ss1 = torch.ops.aten.slice_scatter.default(arg3_1, ss, 1, 0, -1)

    select = torch.ops.aten.select.int(view_2, 1, -1)
    slice_3 = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807)
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, 2, 0, 257)
    select_1 = torch.ops.aten.select.int(ss1, 1, -1)
    slice_5 = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
    copy_1 = torch.ops.aten.copy.default(slice_5, slice_4)
    ss2 = torch.ops.aten.slice_scatter.default(select_1, copy_1, 2, 256, 9223372036854775807)
    sel_ss = torch.ops.aten.select_scatter.default(ss1, ss2, 1, -1)

    slice_6 = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
    slice_7 = torch.ops.aten.slice.Tensor(slice_6, 3, 257, 9223372036854775807)
    slice_8 = torch.ops.aten.slice.Tensor(sel_ss, 1, 1, 9223372036854775807)
    slice_9 = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 256)
    copy_2 = torch.ops.aten.copy.default(slice_9, slice_7)
    ss3 = torch.ops.aten.slice_scatter.default(slice_8, copy_2, 3, 0, 256)
    ss4 = torch.ops.aten.slice_scatter.default(sel_ss, ss3, 1, 1, 9223372036854775807)

    select_2 = torch.ops.aten.select.int(view_2, 1, 0)
    slice_10 = torch.ops.aten.slice.Tensor(select_2, 1, 0, 255)
    slice_11 = torch.ops.aten.slice.Tensor(slice_10, 2, -255, 9223372036854775807)
    select_3 = torch.ops.aten.select.int(ss4, 1, 0)
    slice_12 = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
    slice_13 = torch.ops.aten.slice.Tensor(slice_12, 2, 1, 256)
    copy_3 = torch.ops.aten.copy.default(slice_13, slice_11)
    ss5 = torch.ops.aten.slice_scatter.default(slice_12, copy_3, 2, 1, 256)
    ss6 = torch.ops.aten.slice_scatter.default(select_3, ss5, 1, 1, 256)
    sel_ss1 = torch.ops.aten.select_scatter.default(ss4, ss6, 1, 0)

    view_3 = torch.ops.aten.view.default(sel_ss1, [8, 12, 1024, 513])
    p1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3])

    # ---- Row-wise mask fill (query start/end) -----------------------------
    slice_14 = torch.ops.aten.slice.Tensor(p1, 1, 0, 256)
    slice_15 = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 257)
    where_ = torch.ops.aten.where.self(arg4_1, arg5_1, slice_15)
    copy_4 = torch.ops.aten.copy.default(slice_15, where_)
    ss7 = torch.ops.aten.slice_scatter.default(slice_14, copy_4, 3, 0, 257)
    ss8 = torch.ops.aten.slice_scatter.default(p1, ss7, 1, 0, 256)
    p2 = torch.ops.aten.permute.default(ss8, [0, 2, 1, 3])
    v4 = torch.ops.aten.view.default(p2, [96, 4, 256, 513])
    v5 = torch.ops.aten.view.default(v4, [8, 12, 1024, 513])
    p3 = torch.ops.aten.permute.default(v5, [0, 2, 1, 3])

    slice_16 = torch.ops.aten.slice.Tensor(p3, 1, -256, 9223372036854775807)
    slice_17 = torch.ops.aten.slice.Tensor(slice_16, 3, -257, 9223372036854775807)
    where_1 = torch.ops.aten.where.self(arg6_1, arg5_1, slice_17)
    copy_5 = torch.ops.aten.copy.default(slice_17, where_1)
    ss9 = torch.ops.aten.slice_scatter.default(slice_16, copy_5, 3, -257, 9223372036854775807)
    ss10 = torch.ops.aten.slice_scatter.default(p3, ss9, 1, -256, 9223372036854775807)
    p4 = torch.ops.aten.permute.default(ss10, [0, 2, 1, 3])
    p5 = torch.ops.aten.permute.default(p4, [0, 2, 1, 3])
    add_v = torch.ops.aten.add.Tensor(p5, arg7_1)
    p6 = torch.ops.aten.permute.default(add_v, [0, 2, 1, 3])
    p7 = torch.ops.aten.permute.default(p6, [0, 2, 1, 3])
    return p7


def _final_layout(mul_1_bf16, device):
    """Reproduce the eager tail: mul_1 -> view_9 = bf16[384, 256, 768]."""
    p8 = torch.ops.aten.permute.default(mul_1_bf16, [0, 2, 1, 3])
    c1 = torch.ops.aten.clone.default(p8, memory_format=torch.contiguous_format)
    v6 = torch.ops.aten.view.default(c1, [96, 4, 256, 513])
    pad1 = torch.ops.aten.constant_pad_nd.default(v6, [0, 257], 0.0)
    v7 = torch.ops.aten.view.default(pad1, [96, 4, -1])
    slice_18 = torch.ops.aten.slice.Tensor(v7, 2, 0, -256)
    v8 = torch.ops.aten.view.default(slice_18, [96, 4, 256, 769])
    slice_19 = torch.ops.aten.slice.Tensor(v8, 3, 0, -1)
    unsq = torch.ops.aten.unsqueeze.default(slice_19, 4)
    v9 = torch.ops.aten.view.default(unsq, [384, 256, 768])
    return v9


@oracle_impl(hardware="B200", point="b64f0e8a")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1, *_shape_params,
    ) = inputs
    del _shape_params

    device = arg0_1.device

    # ---- Reproduce eager permute_7 (bf16[8, 1024, 12, 513]) --------------
    permute_7 = _build_permute_7(
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, device,
    )

    # ---- Pre-generate inductor_random (dropout mask) ---------------------
    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random_shape = (BATCH, SEQ, HEADS, WINDOW)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # ---- Prepare kernel inputs -------------------------------------------
    perm7_2d = permute_7.contiguous().view(ROWS, WINDOW)
    random_2d = random.contiguous().view(ROWS, WINDOW)

    perm7_padded = torch.zeros(ROWS, BLOCK_N, device=device, dtype=torch.bfloat16)
    random_padded = torch.zeros(ROWS, BLOCK_N, device=device, dtype=torch.float32)
    perm7_padded[:, :WINDOW] = perm7_2d
    random_padded[:, :WINDOW] = random_2d

    # arg8_1: bool[8, 1024, 1, 1] broadcasts to [8, 1024, 12, 513] via [b,s,:,:].
    # In row-major flatten [8,1024,12,513] -> row r = b*(SEQ*HEADS) + s*HEADS + h.
    # arg8_1's (b, s) doesn't depend on h/col, so we tile it.
    query_mask = (
        arg8_1.view(BATCH, SEQ)
              .unsqueeze(-1).expand(BATCH, SEQ, HEADS)
              .contiguous()
              .view(-1)
    )

    fill_1d = arg9_1.view(1).contiguous()

    valid_mask = torch.zeros(BLOCK_N, device=device, dtype=torch.bool)
    valid_mask[:WINDOW] = True

    amax_flat = torch.empty(ROWS, device=device, dtype=torch.float32)
    sum_flat = torch.empty(ROWS, device=device, dtype=torch.float32)
    keep_padded = torch.empty(ROWS, BLOCK_N, device=device, dtype=torch.bool)
    mul_out_padded = torch.empty(ROWS, BLOCK_N, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _softmax_dropout_kernel,
        (perm7_padded, query_mask, fill_1d, random_padded,
         amax_flat, sum_flat, keep_padded, mul_out_padded,
         valid_mask, BLOCK_N),
    )

    # Slice kernel outputs back to WINDOW / reshape to [8,1024,12,...] shape.
    amax = amax_flat.view(BATCH, SEQ, HEADS, 1)
    sum_1 = sum_flat.view(BATCH, SEQ, HEADS, 1)
    gt = keep_padded[:, :WINDOW].contiguous().view(BATCH, SEQ, HEADS, WINDOW)
    mul_1_bf16 = mul_out_padded[:, :WINDOW].contiguous().view(BATCH, SEQ, HEADS, WINDOW)

    view_9 = _final_layout(mul_1_bf16, device)
    permute_9 = torch.ops.aten.permute.default(view_9, [0, 2, 1])

    return permute_7, amax, sum_1, gt, view_9, permute_9
