"""cuTile port of amax_sum_e1a43ba9dec2: T5 attention softmax + Inductor dropout.

The score bias is strided (f32[8,8,1024,1024], stride=[8388608,1,8192,8]) —
we materialize a dense view before feeding the row kernel. Then softmax with
amax/sum side outputs and Inductor-seeded dropout happen in one row kernel.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 57
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    scores_bf_ptr,   # bf16 [n_rows, k_len]
    random_ptr,      # f32  [n_rows, k_len]
    amax_ptr,        # f32  [n_rows]
    sum_ptr,         # f32  [n_rows]
    gt_ptr,          # b8   [n_rows, k_len]
    dropped_ptr,     # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    scores_bf = ct.load(scores_bf_ptr, index=(row, 0), shape=(1, K_LEN))
    scores = ct.astype(scores_bf, ct.float32)
    row_max = ct.max(scores)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer)
    probs_bf = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), row_max, dtype=ct.float32), (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), denom, dtype=ct.float32), (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, K_LEN), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(
        ct.where(keep, ct.astype(probs_bf, ct.float32), 0.0),
        ct.bfloat16,
    )
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="aeb1682d")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, view_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    device = arg0_1.device
    full_shape = _as_shape(view_shape_arg)
    random_shape = _as_shape(random_shape_arg)
    out_shape = _as_shape(out_shape_arg)
    k_len = int(full_shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    # View arg0_1 [64,1024,1024] as [8,8,1024,1024], then add strided f32 bias
    # arg1_1. Materialize the result as a dense bf16 tensor for the kernel.
    x_view = arg0_1.view(full_shape)
    scores_full_f32 = x_view.float() + arg1_1
    scores_full_bf = scores_full_f32.to(torch.bfloat16)
    scores_2d = scores_full_bf.contiguous().view(n_rows, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len).contiguous()

    amax_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)
    sum_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)
    gt_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bool)
    dropped_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (scores_2d, random_2d, amax_1d, sum_1d, gt_2d, dropped_2d, k_len),
    )

    rounded = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16,
    )
    rounded.view(n_rows, k_len).copy_(scores_2d)
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    amax.view(n_rows).copy_(amax_1d)
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    sum_1.view(n_rows).copy_(sum_1d)
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool,
    )
    gt.view(n_rows, k_len).copy_(gt_2d)
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )
    dropped.view(n_rows, k_len).copy_(dropped_2d)

    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
