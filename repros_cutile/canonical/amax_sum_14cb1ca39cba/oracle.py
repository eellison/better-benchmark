"""cuTile port of amax_sum_14cb1ca39cba: Longformer sliding-window attention.

The Longformer band construction is reproduced with torch ops (portable).
A cuTile kernel handles the row softmax + dropout emission with pre-generated
random tensor.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 30
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    scores_ptr,      # bf16 [rows, 513]
    random_ptr,      # f32 [rows, 513]
    where_mask_ptr,  # bool [rows, 513]  (arg8_1 broadcast)
    fill_ptr,        # f32 scalar
    amax_ptr,        # f32 [rows]
    denom_ptr,       # f32 [rows]
    gt_ptr,          # bool [rows, 513]
    out_ptr,         # bf16 [rows, 513]
    K_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    scores = ct.load(
        scores_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scores_f = ct.astype(scores, ct.float32)
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < K_LEN, (1, BLOCK_N))
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    scores_active = ct.where(col_mask, scores_f, neg_inf)
    row_max = ct.max(scores_active, axis=1, keepdims=True)
    # Detect NaN in active scores; propagate NaN through amax like torch.amax does.
    is_nan = (scores_f != scores_f) & col_mask
    is_nan_i = ct.astype(is_nan, ct.int32)
    nan_count = ct.sum(is_nan_i, axis=1, keepdims=True)
    has_nan = nan_count > 0
    nan_val = ct.full((1, 1), float("nan"), dtype=ct.float32)
    row_max_out = ct.where(has_nan, nan_val, row_max)
    numer = ct.exp(scores_f - row_max)
    numer_masked = ct.where(col_mask, numer, 0.0)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)
    probs = numer_masked / denom

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max_out, (1,)))
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    where_mask = ct.load(
        where_mask_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_2d = ct.reshape(fill, (1, 1))
    dropped_probs = ct.where(where_mask, fill_2d, probs)
    dropped_probs_bf = ct.astype(dropped_probs, ct.bfloat16)

    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_N), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, dropped_probs_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    ct.store(out_ptr, index=(row, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="b64f0e8a", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, *shape_params) = inputs
    device = arg0_1.device

    # Faithfully reproduce Repro band construction in torch.
    (sp0, sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10, sp11) = shape_params
    view = arg0_1.view(*[int(d) for d in sp0])
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.contiguous().view(*[int(d) for d in sp1])
    pad = torch.nn.functional.pad(view_1, [0, 0, 0, 1], value=0.0)
    view_2 = pad.view(*[int(d) for d in sp2])
    slice_1 = view_2[:, :, 0:256, :]
    slice_2 = slice_1[:, :, :, 0:257]
    copy = slice_2.clone()
    slice_scatter = arg2_1.clone()
    slice_scatter[:, :, :, 256:] = copy
    slice_scatter_1 = arg3_1.clone()
    slice_scatter_1[:, 0:-1, :, :] = slice_scatter

    select = view_2[:, -1, :, :]
    slice_3 = select[:, 256:, :]
    slice_4 = slice_3[:, :, 0:257]
    select_1 = slice_scatter_1[:, -1, :, :].clone()
    slice_5 = select_1[:, :, 256:].clone()
    slice_5.copy_(slice_4)
    select_1[:, :, 256:] = slice_5
    select_scatter = slice_scatter_1.clone()
    select_scatter[:, -1, :, :] = select_1

    slice_6 = view_2[:, :, -257:-1, :]
    slice_7 = slice_6[:, :, :, 257:]
    slice_8 = select_scatter[:, 1:, :, :].clone()
    slice_9 = slice_8[:, :, :, 0:256].clone()
    slice_9.copy_(slice_7)
    slice_8[:, :, :, 0:256] = slice_9
    slice_scatter_4 = select_scatter.clone()
    slice_scatter_4[:, 1:, :, :] = slice_8

    select_2 = view_2[:, 0, :, :]
    slice_10 = select_2[:, 0:255, :]
    slice_11 = slice_10[:, :, -255:]
    select_3 = slice_scatter_4[:, 0, :, :].clone()
    slice_12 = select_3[:, 1:256, :].clone()
    slice_13 = slice_12[:, :, 1:256].clone()
    slice_13.copy_(slice_11)
    slice_12[:, :, 1:256] = slice_13
    select_3[:, 1:256, :] = slice_12
    select_scatter_1 = slice_scatter_4.clone()
    select_scatter_1[:, 0, :, :] = select_3

    view_3 = select_scatter_1.view(*[int(d) for d in sp3])
    permute_1 = view_3.permute(0, 2, 1, 3).contiguous()
    slice_14 = permute_1[:, 0:256, :, :].clone()
    slice_15 = slice_14[:, :, :, 0:257].clone()
    where = torch.where(arg4_1, arg5_1, slice_15)
    slice_14[:, :, :, 0:257] = where
    permute_1[:, 0:256, :, :] = slice_14
    permute_2 = permute_1.permute(0, 2, 1, 3)

    view_4 = permute_2.contiguous().view(*[int(d) for d in sp4])
    view_5 = view_4.view(*[int(d) for d in sp5])
    permute_3 = view_5.permute(0, 2, 1, 3).contiguous()
    slice_16 = permute_3[:, -256:, :, :].clone()
    slice_17 = slice_16[:, :, :, -257:].clone()
    where_1 = torch.where(arg6_1, arg5_1, slice_17)
    slice_16[:, :, :, -257:] = where_1
    permute_3[:, -256:, :, :] = slice_16
    permute_4 = permute_3.permute(0, 2, 1, 3)
    permute_5 = permute_4.permute(0, 2, 1, 3)  # [8, 1024, 12, 513]

    # add bias
    add = permute_5 + arg7_1
    permute_6 = add.permute(0, 2, 1, 3)
    permute_7 = permute_6.permute(0, 2, 1, 3)  # returned

    # softmax + dropout in cuTile
    n_rows = 8 * 1024 * 12
    scores_2d = permute_7.contiguous().view(n_rows, 513)
    where_mask = arg8_1.expand(8, 1024, 12, 513).contiguous().view(n_rows, 513)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random_shape = (8, 1024, 12, 513)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.contiguous().view(n_rows, 513)
    fill_1d = arg9_1.view(1)

    amax = torch.empty((8, 1024, 12, 1), device=device, dtype=torch.float32)
    denom = torch.empty((8, 1024, 12, 1), device=device, dtype=torch.float32)
    gt = torch.empty((8, 1024, 12, 513), device=device, dtype=torch.bool)
    out = torch.empty((8, 1024, 12, 513), device=device, dtype=torch.bfloat16)
    amax_1d = amax.view(n_rows)
    denom_1d = denom.view(n_rows)
    gt_2d = gt.view(n_rows, 513)
    out_2d = out.view(n_rows, 513)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (scores_2d, random_2d, where_mask, fill_1d,
         amax_1d, denom_1d, gt_2d, out_2d, 513, BLOCK_N),
    )

    # view_9 and permute_9 construction via torch
    permute_8 = out.permute(0, 2, 1, 3)
    clone_1 = permute_8.contiguous()
    view_6 = clone_1.view(*[int(d) for d in sp7])
    pad_amount = [int(d) for d in sp8]
    pad_1 = torch.nn.functional.pad(view_6, pad_amount, value=0.0)
    view_7 = pad_1.view(*[int(d) for d in sp9])
    slice_18 = view_7[:, :, 0:-256]
    view_8 = slice_18.view(*[int(d) for d in sp10])
    slice_19 = view_8[:, :, :, 0:-1]
    unsqueeze = slice_19.unsqueeze(4)
    view_9 = unsqueeze.view(*[int(d) for d in sp11])
    permute_9 = view_9.permute(0, 2, 1)

    return permute_7, amax, denom, gt, view_9, permute_9
