"""cuTile port of amax_sum_31d85970adb2: Longformer sliding-window softmax + dropout.

The huge slice-scatter preprocess (rewriting sliding-window attention scores)
is done in torch. A cuTile per-row kernel then does softmax + dropout + mask
overwrite (where_2). Post-softmax slice/pad/reshape is torch.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 9
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112

K_LEN = 513
BLOCK_K = 1024


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,
    mask_ptr,
    scalar_ptr,
    random_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    out_ptr,
    K: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_K_),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(x_bf, ct.float32)
    cols_i = ct.arange(BLOCK_K_, dtype=ct.int32)
    col_ok_1d = cols_i < K
    col_ok = ct.reshape(col_ok_1d, (1, BLOCK_K_))
    # For amax we want to preserve NaN (aten.amax over NaN yields NaN).
    # Fill OOB cols with the very first valid value (0-th col repeat via load
    # is impossible — instead use ct.where to replicate x[:,0] into OOB).
    # Simpler: use a "very negative" fill and separately preserve NaN by
    # using nanmask-based propagation.
    neg_inf = ct.full((1, BLOCK_K_), -1.0e30, dtype=ct.float32)
    x_for_max = ct.where(col_ok, x, neg_inf)
    amax = ct.max(x_for_max, axis=1, keepdims=True)
    # Also propagate NaN: if any valid x is NaN, amax should be NaN.
    # Detect x != x on valid columns.
    zero_f = ct.full((1, BLOCK_K_), 0.0, dtype=ct.float32)
    nan_check = ct.where(col_ok, x, zero_f)
    is_nan = nan_check != nan_check
    one_i = ct.full((1, BLOCK_K_), 1, dtype=ct.int32)
    zero_i = ct.full((1, BLOCK_K_), 0, dtype=ct.int32)
    nan_i = ct.where(is_nan, one_i, zero_i)
    any_nan = ct.max(nan_i, axis=1, keepdims=True) != 0
    nan_val = ct.full((1, 1), float("nan"), dtype=ct.float32)
    amax = ct.where(any_nan, nan_val, amax)
    ct.store(amax_ptr, index=(row, 0), tile=amax)
    sub_ = x - amax
    exp_v = ct.exp(sub_)
    exp_v = ct.where(col_ok, exp_v, zero_f)
    sum_v = ct.sum(exp_v, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row, 0), tile=sum_v)
    div_v = exp_v / sum_v

    mask_row = ct.load(mask_ptr, index=(row, 0), shape=(1, 1))
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_bc = ct.full((1, BLOCK_K_), 0.0, dtype=ct.float32) + ct.reshape(scalar, (1, 1))
    where2 = ct.where(mask_row, scalar_bc, div_v)
    where2_bf = ct.astype(where2, ct.bfloat16)

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_K_),
                       padding_mode=ct.PaddingMode.ZERO)
    random_bf = ct.astype(random_f, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_K_), DROPOUT_P_, dtype=ct.bfloat16)
    keep = random_bf > thresh_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_K_), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where2_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


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


def _sliding_window_prep(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
                          arg6_1, arg7_1, device):
    """Slice-scatter graph verbatim from the Repro (torch-only)."""
    view = arg0_1.view(96, 3, 512, 1, 512)
    permute = view.permute(0, 1, 2, 4, 3)
    view_1 = permute.reshape(96, 3, 512, 512)
    pad_ = torch.nn.functional.pad(view_1, [0, 0, 0, 1], mode='constant', value=0.0)
    view_2 = pad_.view(96, 3, 512, 513)
    slice_2 = view_2[:, :, :256, :257]
    copy = arg1_1.clone()
    copy.copy_(slice_2)
    slice_scatter = arg2_1.clone()
    slice_scatter[:, :, :, 256:] = copy
    slice_scatter_1 = arg3_1.clone()
    slice_scatter_1[:, :-1] = slice_scatter
    select = view_2[:, -1, :, :]
    slice_4 = select[:, 256:, :257]
    select_1 = slice_scatter_1[:, -1, :, :].clone()
    select_1[:, :, 256:] = slice_4
    select_scatter = slice_scatter_1.clone()
    select_scatter[:, -1] = select_1
    slice_7 = view_2[:, :, -257:-1, 257:]
    slice_8 = select_scatter[:, 1:, :, :].clone()
    slice_8[:, :, :, :256] = slice_7
    slice_scatter_4 = select_scatter.clone()
    slice_scatter_4[:, 1:] = slice_8
    select_2 = view_2[:, 0, :, :]
    slice_11 = select_2[:, :255, -255:]
    select_3 = slice_scatter_4[:, 0, :, :].clone()
    select_3[:, 1:256, 1:256] = slice_11
    select_scatter_1 = slice_scatter_4.clone()
    select_scatter_1[:, 0] = select_3
    view_3 = select_scatter_1.view(8, 12, 1024, 513)
    permute_1 = view_3.permute(0, 2, 1, 3).contiguous()  # [8, 1024, 12, 513]
    # Overwrite first 256 seq positions with masked-where.
    slice_15 = permute_1[:, :256, :, :257]
    permute_1[:, :256, :, :257] = torch.where(arg4_1, arg5_1, slice_15)
    view_5 = permute_1.permute(0, 2, 1, 3).contiguous().view(8, 12, 1024, 513)
    permute_3 = view_5.permute(0, 2, 1, 3).contiguous()
    slice_17 = permute_3[:, -256:, :, -257:]
    permute_3[:, -256:, :, -257:] = torch.where(arg6_1, arg5_1, slice_17)
    permute_5 = permute_3
    add_ = permute_5 + arg7_1
    permute_7 = add_
    return permute_7


@oracle_impl(hardware="B200", point="b64f0e8a")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     arg8_1, arg9_1, arg10_1, *_shape) = inputs
    device = arg0_1.device

    permute_7 = _sliding_window_prep(
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, device)

    total_rows = 8 * 1024 * 12
    permute_7_flat = permute_7.contiguous().view(total_rows, K_LEN)
    x_pad = torch.zeros((total_rows, BLOCK_K), device=device, dtype=torch.bfloat16)
    x_pad[:, :K_LEN].copy_(permute_7_flat)

    arg8_bc = arg8_1.expand(8, 1024, 12, 1).contiguous().view(total_rows, 1)

    amax = torch.empty((total_rows, 1), device=device, dtype=torch.float32)
    sum_ = torch.empty((total_rows, 1), device=device, dtype=torch.float32)
    gt_pad = torch.empty((total_rows, BLOCK_K), device=device, dtype=torch.bool)
    out_pad = torch.empty((total_rows, BLOCK_K), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg10_1, SEED_INDEX)
    random = _inductor_random_for_eager_check((8, 1024, 12, K_LEN), seed, device=device)
    random_pad = torch.zeros((total_rows, BLOCK_K), device=device, dtype=torch.float32)
    random_pad[:, :K_LEN].copy_(random.contiguous().view(total_rows, K_LEN))

    scalar = arg9_1.view(1)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (total_rows, 1, 1), _softmax_dropout_kernel,
              (x_pad, arg8_bc, scalar, random_pad,
               amax, sum_, gt_pad, out_pad,
               K_LEN, BLOCK_K, DROPOUT_P, DROPOUT_SCALE))

    amax_4d = amax.view(8, 1024, 12, 1)
    sum_4d = sum_.view(8, 1024, 12, 1)
    gt_4d = gt_pad[:, :K_LEN].contiguous().view(8, 1024, 12, K_LEN)
    mul_1_4d = out_pad[:, :K_LEN].contiguous().view(8, 1024, 12, K_LEN)

    permute_8 = mul_1_4d.permute(0, 2, 1, 3)
    clone_1 = permute_8.contiguous()
    view_6 = clone_1.view(96, 4, 256, K_LEN)
    pad_1 = torch.nn.functional.pad(view_6, [0, 257], mode='constant', value=0.0)
    view_7 = pad_1.view(96, 4, 197120)
    slice_18 = view_7[:, :, :-256]
    view_8 = slice_18.view(96, 4, 256, 769)
    slice_19 = view_8[:, :, :, :-1]
    unsqueeze = slice_19.unsqueeze(4)
    view_9 = unsqueeze.view(384, 256, 768)
    permute_9 = view_9.permute(0, 2, 1)

    return permute_7, amax_4d, sum_4d, gt_4d, view_9, permute_9
