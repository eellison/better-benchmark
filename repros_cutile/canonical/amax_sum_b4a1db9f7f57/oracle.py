"""cuTile port of amax_sum_b4a1db9f7f57: MT5 additive-bias attention softmax + dropout.

Pre-generates the seeded random tensor via inductor_random outside the kernel,
then runs a single cuTile row kernel. The bias tensor arg1_1 has stride
[98304, 1, 768, 6] and gets accessed via computed offsets.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 65
ROWS = 32 * 6 * 128
K_LEN = 128
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    arg0_ptr,        # bf16 [ROWS, K_LEN]
    bias_ptr,        # f32 flat storage (already permuted-view src)
    random_ptr,      # f32 [ROWS, K_LEN]
    scores_out,      # bf16 [ROWS, K_LEN]
    amax_out,        # f32 [ROWS]
    sum_out,         # f32 [ROWS]
    gt_out,          # b8 [ROWS, K_LEN]
    value_out,       # bf16 [ROWS, K_LEN]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    # Decompose row into (batch, head, query).
    bh = row // 128
    batch = bh // 6
    head = bh - batch * 6
    query = row - bh * 128

    raw = ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_N))
    raw_f = ct.astype(raw, ct.float32)

    # Bias gather: batch*98304 + head + query*768 + col*6, with col in [0..127]
    cols = ct.arange(BLOCK_N, dtype=ct.int64)
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    base_scalar = batch * 98304 + head + query * 768
    bias_offsets = cols_2d * 6 + base_scalar
    bias_gather = ct.gather(bias_ptr, bias_offsets)
    bias_f = ct.astype(bias_gather, ct.float32)

    rounded_bf = ct.astype(raw_f + bias_f, ct.bfloat16)
    ct.store(scores_out, index=(row, 0), tile=rounded_bf)

    scores = ct.astype(rounded_bf, ct.float32)
    # NaN detection: has_nan = any(scores != scores)
    is_nan = scores != scores
    nan_i = ct.where(is_nan, ct.full((1, BLOCK_N), 1, dtype=ct.int32),
                     ct.zeros((1, BLOCK_N), dtype=ct.int32))
    has_nan = ct.sum(nan_i) > 0

    row_max = ct.max(scores)
    row_max_final = ct.where(has_nan, float("nan"), row_max)
    shifted = scores - row_max_final
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_out, index=(row, 0), tile=keep)

    ct.store(amax_out, index=(row,), tile=ct.reshape(row_max_final, (1,)))
    ct.store(sum_out, index=(row,), tile=ct.reshape(denom, (1,)))

    dropped = ct.astype(ct.astype(probs_bf, ct.float32) * ct.astype(keep, ct.float32), ct.bfloat16)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(value_out, index=(row, 0), tile=scaled)


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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, *_shape_params = inputs
    device = arg0_1.device
    score_shape = (32, 6, 128, 128)
    row_shape = (32, 6, 128, 1)
    out_shape = (192, 128, 128)

    scores = torch.empty_strided(
        score_shape, _contiguous_stride(score_shape),
        device=device, dtype=torch.bfloat16)
    amax_out = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    sum_out = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    gt_out = torch.empty_strided(
        score_shape, _contiguous_stride(score_shape),
        device=device, dtype=torch.bool)
    value_out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(score_shape, seed, device=device)

    scores_2d = arg0_1.contiguous().view(ROWS, K_LEN)
    # bias arg1_1 is f32 [32,6,128,128] but stored with stride [98304, 1, 768, 6].
    # Access via raw storage — use as_strided to get 1D flat view of underlying storage.
    # But we need actual flat storage. Use .as_strided((numel_of_storage,), (1,))
    bias_flat = torch.as_strided(arg1_1, (arg1_1.untyped_storage().size() // arg1_1.element_size(),), (1,))
    random_2d = random.contiguous().view(ROWS, K_LEN)
    scores_out_2d = scores.view(ROWS, K_LEN)
    amax_1d = amax_out.view(ROWS)
    sum_1d = sum_out.view(ROWS)
    gt_2d = gt_out.view(ROWS, K_LEN)
    value_2d = value_out.view(ROWS, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _softmax_dropout_kernel,
        (scores_2d, bias_flat, random_2d,
         scores_out_2d, amax_1d, sum_1d, gt_2d, value_2d,
         BLOCK_N),
    )
    return scores, amax_out, sum_out, gt_out, value_out, value_out.permute(0, 2, 1)
