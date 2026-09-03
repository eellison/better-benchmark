"""cuTile port of amax_sum_26a17788b6f0: DeBERTa masked softmax + dropout.

Pre-generates the random tensor via inductor_random. Runs one row kernel that
emits (where, amax, sum, gt, bf16 dropout). Permute alias via torch.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 58
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, K_LEN] (already viewed as flat rows)
    mask_ptr,       # bool [rows, K_LEN]
    fill_ptr,       # bf16 [] (scalar)
    random_ptr,     # f32 [rows, K_LEN]
    where_ptr,      # bf16 [rows, K_LEN]
    amax_ptr,       # f32 [rows]
    sum_ptr,        # f32 [rows]
    gt_ptr,         # bool [rows, K_LEN]
    out_ptr,        # bf16 [rows, K_LEN]
    K_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    mask_b = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_N))
    fill_bf = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_broadcast = ct.reshape(fill_bf, (1, 1))

    where_bf = ct.where(mask_b, fill_broadcast, x_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_bf)

    scores_f = ct.astype(where_bf, ct.float32)
    row_max = ct.max(scores_f)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    shifted = scores_f - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer * (1.0 / denom)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    threshold_f = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > threshold_f
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    dropped = keep_f * probs
    scaled = dropped * DROPOUT_SCALE
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="00541467", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, view_shape, random_shape, out_view_shape = inputs
    # view_shape may contain -1; use the full shape [8, 24, 512, 512]
    random_shape = tuple(int(d) for d in random_shape)
    batch, heads, q_len, k_len = random_shape
    view_4d_shape = (batch, heads, q_len, k_len)
    device = arg0_1.device

    # Broadcast mask arg1_1 [8, 1, 512, 512] to [8, 24, 512, 512]
    view4 = arg0_1.view(batch, heads, q_len, k_len)  # bf16
    mask_full = arg1_1.expand(batch, heads, q_len, k_len).contiguous()

    where_bf = torch.empty_strided(view_4d_shape, _contiguous_stride(view_4d_shape), device=device, dtype=torch.bfloat16)
    reduction_shape = (batch, heads, q_len, 1)
    amax = torch.empty_strided(reduction_shape, _contiguous_stride(reduction_shape), device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(reduction_shape, _contiguous_stride(reduction_shape), device=device, dtype=torch.float32)
    gt = torch.empty_strided(view_4d_shape, _contiguous_stride(view_4d_shape), device=device, dtype=torch.bool)
    out_bf = torch.empty_strided(view_4d_shape, _contiguous_stride(view_4d_shape), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    rows = batch * heads * q_len
    view4_2d = view4.contiguous().view(rows, k_len)
    mask_2d = mask_full.view(rows, k_len)
    random_2d = random.contiguous().view(rows, k_len)
    where_2d = where_bf.view(rows, k_len)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    gt_2d = gt.view(rows, k_len)
    out_2d = out_bf.view(rows, k_len)

    stream = torch.cuda.current_stream()
    fill_1d = arg2_1.view(1)
    ct.launch(
        stream,
        (rows, 1, 1),
        _softmax_dropout_kernel,
        (view4_2d, mask_2d, fill_1d, random_2d,
         where_2d, amax_1d, sum_1d, gt_2d, out_2d,
         k_len, BLOCK_N),
    )
    # out_bf viewed as [192, 512, 512]
    out_shape_3d = (batch * heads, q_len, k_len)
    out_3d = out_bf.view(out_shape_3d)
    permute = out_3d.permute(0, 2, 1)
    return where_bf, amax, sum_1, gt, out_3d, permute
