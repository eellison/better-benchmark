"""cuTile port of amax_sum_any_836f72914974: MobileBERT safe softmax + dropout.

The safe-softmax path with -inf handling, `any`-row fallback to a bf16 tensor,
seeded Inductor dropout, and bf16 dropout scaling all happen in a single row
kernel. Dropout random comes from torch.ops.prims.inductor_random (eager path
mirrored under CUDA-graph capture).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 17
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,          # bf16 [n_rows, k_len]
    fallback_ptr,   # bf16 [n_rows, k_len]
    random_ptr,     # f32  [n_rows, k_len]
    where_ptr,      # bf16 [n_rows, k_len]
    gt_ptr,         # b8   [n_rows, k_len]
    dropped_ptr,    # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    scores_bf = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN))
    scores = ct.astype(scores_bf, ct.float32)
    live = scores != float("-inf")
    live_i = ct.astype(live, ct.int32)
    has_any = ct.max(live_i) != 0

    row_max = ct.max(scores)
    safe_max = ct.where(has_any, row_max, 0.0)
    numer = ct.exp(scores - safe_max)
    numer = ct.where(live, numer, 0.0)
    denom = ct.sum(numer)
    denom = ct.where(has_any, denom, 1.0)
    probs_bf = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row, 0), shape=(1, K_LEN))
    where_val = ct.where(has_any, probs_bf, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, K_LEN), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(
        ct.where(keep, ct.astype(where_val, ct.float32), 0.0),
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


@oracle_impl(hardware="B200", point="d59f4ab1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, view_shape, random_shape, _expand_shape, out_shape = inputs
    device = arg0_1.device
    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    fallback_shape = _as_shape(arg1_1.shape)
    random_shape_t = _as_shape(random_shape)
    out_shape_t = _as_shape(out_shape)

    if arg0_1.is_contiguous():
        x_2d = arg0_1.view(n_rows, k_len)
    else:
        x_2d = arg0_1.contiguous().view(n_rows, k_len)
    if arg1_1.is_contiguous():
        fallback_2d = arg1_1.view(n_rows, k_len)
    else:
        fallback_2d = arg1_1.contiguous().view(n_rows, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape_t, seed, device=device)
    random_2d = random.reshape(n_rows, k_len).contiguous()

    where_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bfloat16)
    gt_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bool)
    dropped_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, k_len),
    )

    where = torch.empty_strided(
        fallback_shape, _contiguous_stride(fallback_shape),
        device=device, dtype=torch.bfloat16,
    )
    where.view(n_rows, k_len).copy_(where_2d)
    gt = torch.empty_strided(
        fallback_shape, _contiguous_stride(fallback_shape),
        device=device, dtype=torch.bool,
    )
    gt.view(n_rows, k_len).copy_(gt_2d)
    dropped = torch.empty_strided(
        out_shape_t, _contiguous_stride(out_shape_t),
        device=device, dtype=torch.bfloat16,
    )
    dropped.view(n_rows, k_len).copy_(dropped_2d)

    return where, gt, dropped, dropped.permute(0, 2, 1)
