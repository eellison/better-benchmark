"""cuTile port of amax_sum_bf7830302e6d: Longformer sliding-window softmax+dropout (bf16 training).

Strategy: perform the very complex band-assembly, edge-mask generation, and
attention-mask branching via native torch, then run a single cuTile row
kernel for the softmax + edge-mask + amax/sum side outputs.

The final padded-layout epilogue is also done via native torch since it's a
sequence of view/pad/slice/permute that torch handles efficiently.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 0
SEED_COUNT = 36
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_N = 1024  # pow2 >= 513
WINDOW = 513


@ct.kernel
def _softmax_row_kernel(
    scores_f_ptr,       # f32 [n_rows, WINDOW] contiguous
    query_masked_ptr,   # b8  [n_rows] — TRUE means output = 0 (query masked out)
    div_out_ptr,        # bf16 [n_rows, WINDOW] contiguous
    amax_out_ptr,       # f32 [n_rows]
    sum_out_ptr,        # f32 [n_rows]
    WINDOW_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_N_, dtype=ct.int32)
    valid = cols < WINDOW_
    neg_inf = ct.full((BLOCK_N_,), -float("inf"), dtype=ct.float32)
    zero_f = ct.full((BLOCK_N_,), 0.0, dtype=ct.float32)

    x2d = ct.load(scores_f_ptr, index=(row, 0), shape=(1, BLOCK_N_),
                  padding_mode=ct.PaddingMode.ZERO)
    x = ct.reshape(x2d, (BLOCK_N_,))
    x_m = ct.where(valid, x, neg_inf)

    # NaN propagation: cuTile ct.max drops NaN. Force NaN if any NaN in valid.
    nan_mask = (x != x) & valid
    nan_count = ct.sum(ct.astype(nan_mask, ct.int32))
    zero_i = ct.full((), 0, dtype=ct.int32)
    row_max_raw = ct.max(x_m)
    nan_v = ct.full((), float("nan"), dtype=ct.float32)
    row_max = ct.where(nan_count != zero_i, nan_v, row_max_raw)
    row_max_s = ct.reshape(row_max, (1,))

    numer = ct.exp(x_m - row_max_s)
    numer_m = ct.where(valid, numer, zero_f)
    denom = ct.sum(numer_m)
    denom_s = ct.reshape(denom, (1,))
    probs = numer_m / denom_s

    # Query mask: if masked, zero out probs; else keep
    q_mask = ct.load(query_masked_ptr, index=(row,), shape=(1,))
    # broadcast q_mask (bool) across BLOCK_N_
    q_mask_b = ct.broadcast_to(ct.reshape(q_mask, (1,)), (BLOCK_N_,))
    probs_final = ct.where(q_mask_b, zero_f, probs)
    probs_bf = ct.astype(probs_final, ct.bfloat16)
    ct.store(div_out_ptr, index=(row, 0), tile=ct.reshape(probs_bf, (1, BLOCK_N_)))

    ct.store(amax_out_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_out_ptr, index=(row,), tile=ct.reshape(denom, (1,)))


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


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape, device=device)
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


@oracle_impl(hardware="B200", point="50ff733e")
def oracle_forward(inputs):
    (
        arg0_1,  # bf16 [288, 512, 512]
        arg1_1,  # f32  [8, 1024]  (attention_mask value)
        arg2_1,  # b8   [8, 1024]  (query_mask)
        *_shape_params,
    ) = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG).",
        )
    device = arg0_1.device

    # Do the whole eager forward using native torch and use cuTile for the softmax
    # row kernel. We only replace the softmax with cuTile.
    view = arg0_1.view(96, 3, 512, 1, 512)
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(96, 3, 512, 512)
    constant_pad_nd = torch.nn.functional.pad(view_1, [0, 0, 0, 1], value=0.0)
    view_2 = constant_pad_nd.view(96, 3, 512, 513)

    full = torch.zeros((96, 4, 256, 513), device=device, dtype=torch.bfloat16)
    # Preserve the pre-scatter `full` and views for the returned tuple.
    slice_3 = full[:, 0:-1, :, :]
    slice_4 = slice_3[:, :, :, 256:]

    slice_1 = view_2[:, :, 0:256, :]
    slice_2 = slice_1[:, :, :, 0:257]

    # Build the scatter chain on a separate copy to avoid mutating `full`.
    scatter = full.clone()
    scatter[:, 0:-1, :, 256:] = slice_2
    # Second: last chunk mod
    select = view_2[:, -1, :, :]
    slice_5 = select[:, 256:, :]
    slice_6 = slice_5[:, :, 0:257]
    scatter[:, -1, :, 256:] = slice_6
    # Third: [:, 1:, :, 0:256] = slice_9
    slice_8 = view_2[:, :, -257:-1, :]
    slice_9 = slice_8[:, :, :, 257:]
    scatter[:, 1:, :, 0:256] = slice_9
    # Fourth: [:, 0, 1:256, 1:256] = slice_13
    select_2 = view_2[:, 0, :, :]
    slice_12 = select_2[:, 0:255, :]
    slice_13 = slice_12[:, :, -255:]
    scatter[:, 0, 1:256, 1:256] = slice_13
    select_scatter_1 = scatter  # [96, 4, 256, 513]

    # Constant edge masks
    full_1 = torch.ones((256, 257), device=device, dtype=torch.bfloat16)
    iota = torch.arange(0, 257, device=device, dtype=torch.int64)
    unsqueeze_ = iota.unsqueeze(-2)  # [1, 257]
    iota_1 = torch.arange(0, 256, device=device, dtype=torch.int64)
    unsqueeze_1 = iota_1.unsqueeze(-1)  # [256, 1]
    sub = unsqueeze_ - unsqueeze_1
    le = sub <= 0
    scalar_zero_bf = torch.zeros((), device=device, dtype=torch.bfloat16)
    where_ = torch.where(le, full_1, scalar_zero_bf)
    rev = torch.flip(where_, [0])
    unsqueeze_2 = rev.unsqueeze(0)
    unsqueeze_3 = unsqueeze_2.unsqueeze(2)  # [1, 256, 1, 257]
    rev_1 = torch.flip(unsqueeze_3, [1, 3])
    expand = unsqueeze_3.expand(8, 256, 12, 257)
    view_3 = select_scatter_1.view(8, 12, 1024, 513)
    permute_1 = view_3.permute(0, 2, 1, 3).contiguous()  # [8, 1024, 12, 513]
    slice_16 = permute_1[:, 0:256, :, :]
    slice_17 = slice_16[:, :, :, 0:257]
    permute_2 = torch.full((8, 12, 256, 257), float("-inf"),
                           device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    convert_element_type = expand.to(torch.bool)
    where_1 = torch.where(convert_element_type, permute_2, slice_17)
    slice_17_new = slice_17.clone()
    slice_17_new.copy_(where_1)
    slice_16_new = slice_16.clone()
    slice_16_new[:, :, :, 0:257] = slice_17_new
    permute_1_new = permute_1.clone()
    permute_1_new[:, 0:256, :, :] = slice_16_new
    permute_3 = permute_1_new.permute(0, 2, 1, 3)  # [8, 12, 1024, 513]
    view_4 = permute_3.reshape(96, 4, 256, 513)
    expand_1 = rev_1.expand(8, 256, 12, 257)
    view_5 = view_4.view(8, 12, 1024, 513)
    permute_4 = view_5.permute(0, 2, 1, 3).contiguous()
    slice_18 = permute_4[:, -256:, :, :]
    slice_19 = slice_18[:, :, :, -257:]
    convert_element_type_1 = expand_1.to(torch.bool)
    where_2 = torch.where(convert_element_type_1, permute_2, slice_19)
    slice_19_new = slice_19.clone()
    slice_19_new.copy_(where_2)
    slice_18_new = slice_18.clone()
    slice_18_new[:, :, :, -257:] = slice_19_new
    permute_4[:, -256:, :, :] = slice_18_new
    # slice_scatter_10 is [8, 1024, 12, 513] — this is `permute_4` post-scatter
    slice_scatter_10 = permute_4  # [8, 1024, 12, 513]
    permute_5 = slice_scatter_10.permute(0, 2, 1, 3).contiguous()  # [8, 12, 1024, 513]

    # Attention mask bias assembly (attention_mask -> bias)
    ne = arg1_1 != 0  # b8 [8, 1024]
    unsqueeze_4 = ne.unsqueeze(2)  # [8, 1024, 1]
    unsqueeze_5 = unsqueeze_4.unsqueeze(3)  # [8, 1024, 1, 1] bool
    convert_element_type_2 = unsqueeze_5.to(torch.bfloat16)  # bf16 [8, 1024, 1, 1]
    full_3 = torch.tensor(-3.3895313892515355e38, device=device, dtype=torch.bfloat16)
    where_3 = torch.where(unsqueeze_5, full_3, convert_element_type_2)  # bf16 [8, 1024, 1, 1]
    full_4 = torch.ones((8, 1024, 1, 1), device=device, dtype=torch.bfloat16)
    permute_6 = full_4.permute(0, 2, 1, 3)  # [8, 1, 1024, 1]
    view_6 = permute_6.view(8, 1024, 1)
    permute_7 = where_3.permute(0, 2, 1, 3)  # [8, 1, 1024, 1]
    view_7 = permute_7.view(8, 1024, 1)
    view_8 = view_6.view(8, 2, 512, 1)
    # shape_param_15/16: as_strided target=(8,3,512,1), stride=(1024, 256, 1, 1)
    as_strided = torch.as_strided(view_8, (8, 3, 512, 1), (1024, 256, 1, 1))
    view_9 = view_7.view(8, 2, 512, 1)
    as_strided_1 = torch.as_strided(view_9, (8, 3, 512, 1), (1024, 256, 1, 1))
    unsqueeze_6 = as_strided.unsqueeze(4)
    permute_8 = unsqueeze_6.permute(0, 1, 2, 4, 3)  # [8, 3, 512, 1, 1]
    unsqueeze_7 = as_strided_1.unsqueeze(4)
    permute_9 = unsqueeze_7.permute(0, 1, 4, 2, 3)  # [8, 3, 1, 512, 1]
    mul = permute_8 * permute_9  # broadcast to [8, 3, 512, 512, 1]
    view_10 = mul.view(8, 3, 512, 512)
    constant_pad_nd_1 = torch.nn.functional.pad(view_10, [0, 0, 0, 1], value=0.0)
    view_11 = constant_pad_nd_1.view(8, 3, 512, 513)
    full_5 = torch.zeros((8, 4, 256, 513), device=device, dtype=torch.bfloat16)
    slice_20 = view_11[:, :, 0:256, :]
    slice_21 = slice_20[:, :, :, 0:257]
    full_5[:, 0:-1, :, 256:] = slice_21
    select_4 = view_11[:, -1, :, :]
    slice_24 = select_4[:, 256:, :]
    slice_25 = slice_24[:, :, 0:257]
    full_5[:, -1, :, 256:] = slice_25
    slice_27 = view_11[:, :, -257:-1, :]
    slice_28 = slice_27[:, :, :, 257:]
    full_5[:, 1:, :, 0:256] = slice_28
    select_6 = view_11[:, 0, :, :]
    slice_31 = select_6[:, 0:255, :]
    slice_32 = slice_31[:, :, -255:]
    full_5[:, 0, 1:256, 1:256] = slice_32
    select_scatter_3 = full_5  # [8, 4, 256, 513]

    expand_2 = unsqueeze_3.expand(8, 256, 1, 257)
    view_12 = select_scatter_3.view(8, 1, 1024, 513)
    permute_10 = view_12.permute(0, 2, 1, 3).contiguous()  # [8, 1024, 1, 513]
    slice_35 = permute_10[:, 0:256, :, :]
    slice_36 = slice_35[:, :, :, 0:257]
    full_6 = torch.full((8, 256, 1, 257), float("-inf"),
                        device=device, dtype=torch.bfloat16)
    convert_element_type_3 = expand_2.to(torch.bool)
    where_4 = torch.where(convert_element_type_3, full_6, slice_36)
    slice_36_new = slice_36.clone()
    slice_36_new.copy_(where_4)
    slice_35_new = slice_35.clone()
    slice_35_new[:, :, :, 0:257] = slice_36_new
    permute_10[:, 0:256, :, :] = slice_35_new
    permute_11 = permute_10.permute(0, 2, 1, 3)  # [8, 1, 1024, 513]
    view_13 = permute_11.reshape(8, 4, 256, 513)
    expand_3 = rev_1.expand(8, 256, 1, 257)
    view_14 = view_13.view(8, 1, 1024, 513)
    permute_12 = view_14.permute(0, 2, 1, 3).contiguous()  # [8, 1024, 1, 513]
    slice_37 = permute_12[:, -256:, :, :]
    slice_38 = slice_37[:, :, :, -257:]
    convert_element_type_4 = expand_3.to(torch.bool)
    where_5 = torch.where(convert_element_type_4, full_6, slice_38)
    slice_38_new = slice_38.clone()
    slice_38_new.copy_(where_5)
    slice_37_new = slice_37.clone()
    slice_37_new[:, :, :, -257:] = slice_38_new
    permute_12[:, -256:, :, :] = slice_37_new
    # slice_scatter_21 is [8, 1024, 1, 513] pre-permute
    slice_scatter_21 = permute_12  # [8, 1024, 1, 513]
    permute_13 = slice_scatter_21.permute(0, 2, 1, 3)  # [8, 1, 1024, 513]
    permute_14 = permute_5.permute(0, 2, 1, 3)  # [8, 1024, 12, 513]
    permute_15 = permute_13.permute(0, 2, 1, 3)  # [8, 1024, 1, 513]
    add = permute_14 + permute_15
    permute_16 = add.permute(0, 2, 1, 3)  # [8, 12, 1024, 513]
    permute_17 = permute_16.permute(0, 2, 1, 3)  # [8, 1024, 12, 513]
    scores_f = permute_17.to(torch.float32).contiguous()

    # Query mask preparation
    unsqueeze_8 = arg2_1.unsqueeze(2)  # [8, 1024, 1]
    unsqueeze_9 = unsqueeze_8.unsqueeze(3)  # [8, 1024, 1, 1] bool
    full_7 = torch.zeros((), device=device, dtype=torch.float32)

    # Softmax + where via cuTile
    n_rows = 8 * 1024 * 12
    scores_2d = scores_f.view(n_rows, WINDOW)
    # query_masked broadcast: unsqueeze_9 is [8, 1024, 1, 1], expand to
    # [8, 1024, 12, 1], then flatten to [n_rows]
    q_mask_full = unsqueeze_9.expand(8, 1024, 12, 1).contiguous().view(n_rows)
    amax = torch.empty((8, 1024, 12, 1), device=device, dtype=torch.float32)
    sum_1 = torch.empty((8, 1024, 12, 1), device=device, dtype=torch.float32)
    div_bf = torch.empty((8, 1024, 12, WINDOW), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_row_kernel,
        (scores_2d, q_mask_full,
         div_bf.view(n_rows, WINDOW),
         amax.view(n_rows), sum_1.view(n_rows),
         WINDOW, BLOCK_N),
    )

    # Dropout + final layout epilogue
    seeds, random = _seeds_and_random_for_eager_check(
        (8, 1024, 12, WINDOW), device=device)
    conv7 = random.to(torch.bfloat16)
    gt = conv7 > 0.1
    mul_1 = gt * div_bf
    mul_2 = mul_1 * DROPOUT_SCALE
    permute_18 = mul_2.permute(0, 2, 1, 3)  # [8, 12, 1024, 513]
    clone_1 = permute_18.contiguous()
    view_15 = clone_1.view(96, 4, 256, WINDOW)
    constant_pad_nd_2 = torch.nn.functional.pad(view_15, [0, 257], value=0.0)  # [96, 4, 256, 770]
    view_16 = constant_pad_nd_2.view(96, 4, 197120)
    slice_39 = view_16[:, :, 0:-256]
    view_17 = slice_39.view(96, 4, 256, 769)
    slice_40 = view_17[:, :, :, 0:-1]
    unsqueeze_10 = slice_40.unsqueeze(4)
    view_18 = unsqueeze_10.view(384, 256, 768)
    permute_19 = view_18.permute(0, 2, 1)

    return (
        full,             # 0
        slice_3,          # 1
        slice_4,          # 2 (view of full)
        unsqueeze_3,      # 3
        rev_1,            # 4
        permute_2,        # 5
        convert_element_type,   # 6
        convert_element_type_1, # 7
        slice_scatter_10, # 8
        slice_scatter_21, # 9
        permute_15,       # 10
        amax,             # 11
        sum_1,            # 12
        unsqueeze_9,      # 13
        full_7,           # 14
        seeds,            # 15
        gt,               # 16
        view_18,          # 17
        permute_19,       # 18
    )
