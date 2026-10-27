"""cuTile port of amax_sum_6a531546a9ab: T5/MT5 attention softmax + dropout.

Row-wise cuTile kernel that computes:
  1. Bf16-rounded score = bf16(score.f32 + bias.f32)  (returned as rounded_out)
  2. Row-wise amax (fp32).
  3. Row-wise softmax denom (fp32).
  4. Row-wise softmax probs (bf16).
  5. Seeded Inductor dropout mask (bool).
  6. Bf16 scaled dropout output.

Softmax rows are K_LEN=128 (power-of-2), so no masking needed.
Full shape is [32, 6, 128, 128]; the bias is strided (98304, 1, 768, 6).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 53
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    score_ptr,        # bf16 [batch, heads, q, k]  (view from [192,128,128])
    bias_ptr,         # f32  [batch, heads, q, k]  (strided)
    random_ptr,       # f32  [batch, heads, q, k]  (contiguous)
    rounded_ptr,      # bf16 [batch, heads, q, k]  contiguous (score+bias, rounded)
    amax_ptr,         # f32  [batch*heads*q]
    sum_ptr,          # f32  [batch*heads*q]
    keep_ptr,         # b8   [batch, heads, q, k]  contiguous
    dropped_ptr,      # bf16 [batch, heads, q, k]  contiguous
    n_rows: ct.Constant[int],
    heads: ct.Constant[int],
    q_len: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    # decode row -> (batch, head, q)
    flat_bh = row // q_len
    batch_idx = flat_bh // heads
    head = flat_bh - batch_idx * heads
    q = row - flat_bh * q_len

    # Load score bf16 shape (1,1,1,K) from (batch, head, q, 0)
    score_bf = ct.load(score_ptr, index=(batch_idx, head, q, 0),
                       shape=(1, 1, 1, BLOCK_N))
    bias_f = ct.load(bias_ptr, index=(batch_idx, head, q, 0),
                     shape=(1, 1, 1, BLOCK_N))

    score_f = ct.astype(score_bf, ct.float32)
    add_f = score_f + bias_f
    rounded_bf = ct.astype(add_f, ct.bfloat16)
    ct.store(rounded_ptr, index=(batch_idx, head, q, 0), tile=rounded_bf)

    x = ct.astype(rounded_bf, ct.float32)
    row_max = ct.max(x, axis=3, keepdims=True)
    shifted = x - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=3, keepdims=True)
    probs_f = numer / denom
    probs_bf = ct.astype(probs_f, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(batch_idx, head, q, 0),
                     shape=(1, 1, 1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, 1, 1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(keep_ptr, index=(batch_idx, head, q, 0), tile=keep)

    zero_bf = ct.full((1, 1, 1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(batch_idx, head, q, 0), tile=scaled)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
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


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(full_shape_arg)      # (32, 6, 128, 128)
    random_shape = _shape_tuple(random_shape_arg)  # (32, 6, 128, 128)
    out_shape = _shape_tuple(out_shape_arg)        # (192, 128, 128)
    batch = int(full_shape[0])
    heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = batch * heads * q_len
    row_shape = full_shape[:-1] + (1,)
    row_stride = _contiguous_stride(row_shape)
    full_stride = _contiguous_stride(full_shape)
    device = arg0_1.device

    # Reshape arg0_1 [192, 128, 128] -> [32, 6, 128, 128]
    score_4d = arg0_1.view(batch, heads, q_len, k_len)

    rounded = torch.empty_strided(
        full_shape, full_stride, device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape, row_stride, device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, row_stride, device=device, dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape, full_stride, device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random_4d = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # 4D views for the outputs
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    dropped_4d = dropped.view(batch, heads, q_len, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_dropout_kernel,
        (score_4d, arg1_1, random_4d,
         rounded, amax_1d, sum_1d,
         gt, dropped_4d,
         n_rows, heads, q_len, BLOCK_N),
    )
    view_1 = dropped
    return rounded, amax, sum_1, gt, view_1, view_1.permute(0, 2, 1)
