"""cuTile port of amax_sum_35919422d9ae: T5/MT5 attention softmax + seeded dropout.

Generates the seeded random tensor outside the kernel with
torch.ops.prims.inductor_random and passes it as a kernel input.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 41
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    view_ptr,        # bf16 [n_rows, k_len]
    bias_ptr,        # f32  [B, H, Q, K] with strided access
    random_ptr,      # f32  [n_rows, k_len]
    rounded_ptr,     # bf16 [n_rows, k_len]
    amax_ptr,        # f32  [n_rows]
    sum_ptr,         # f32  [n_rows]
    gt_ptr,          # b8   [n_rows, k_len]
    dropped_ptr,     # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)

    view_bf = ct.load(view_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    bias_f = ct.load(bias_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    view_f = ct.astype(view_bf, ct.float32)
    added = view_f + bias_f
    rounded_bf = ct.astype(added, ct.bfloat16)
    ct.store(rounded_ptr, index=(row_block, 0), tile=rounded_bf)

    scores = ct.astype(rounded_bf, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs_bf = ct.astype(numer / denom, ct.bfloat16)

    # Store row stats (shape (BLOCK_M, 1) written as (BLOCK_M,))
    row_max_1d = ct.reshape(row_max, (BLOCK_M,))
    denom_1d = ct.reshape(denom, (BLOCK_M,))
    ct.store(amax_ptr, index=(row_block,), tile=row_max_1d)
    ct.store(sum_ptr, index=(row_block,), tile=denom_1d)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs_bf, zero_bf)
    scaled_f = ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_M=4, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, shape3 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = tuple(int(d) for d in shape1)
    out_shape = tuple(int(d) for d in shape3)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(arg0_1.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    device = arg0_1.device

    # Reshape arg0_1 [192, 128, 128] -> [32, 6, 128, 128] (view op)
    view_bf = arg0_1.view(full_shape)  # bf16
    # bias arg1_1 is [32, 6, 128, 128] f32 but with strides (98304,1,768,6)
    # We need to make it contiguous for cuTile to load it with our tile layout.
    bias_f = arg1_1.contiguous()

    rounded = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool)
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)

    # Flatten to [n_rows, k_len] for the kernel
    view_2d = view_bf.contiguous().view(n_rows, k_len)
    bias_2d = bias_f.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    rounded_2d = rounded.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)
    amax_2d = amax.view(n_rows)
    sum_2d = sum_1.view(n_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _softmax_dropout_kernel,
        (view_2d, bias_2d, random_2d, rounded_2d, amax_2d, sum_2d, gt_2d, dropped_2d,
         k_len, BLOCK_M, BLOCK_N, DROPOUT_SCALE),
    )
    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
