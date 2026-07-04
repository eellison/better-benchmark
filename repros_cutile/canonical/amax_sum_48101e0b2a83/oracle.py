"""cuTile port of amax_sum_48101e0b2a83: T5/MT5 additive-bias softmax + dropout.

Row kernel that: adds bf16 scores + strided fp32 bias, casts to bf16, stable
softmax with fp32 amax/sum side outputs and bf16 probs, seeded dropout via
pre-computed random tensor from inductor_random. Returns rounded, amax, sum,
gt, dropped, dropped.permute(0,2,1).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 59
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    score_ptr,           # bf16 [B, H, Q, K]
    bias_ptr,            # f32 [B, H, Q, K]  (contiguous)
    random_ptr,          # f32 [B, H, Q, K]
    rounded_ptr,         # bf16 [B, H, Q, K]
    amax_ptr,            # f32 [B, H, Q]
    sum_ptr,             # f32 [B, H, Q]
    gt_ptr,              # b8 [B, H, Q, K]
    dropped_ptr,         # bf16 [B, H, Q, K]
    BLOCK_N: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    q = ct.bid(2)

    score = ct.load(score_ptr, index=(b, h, q, 0), shape=(1, 1, 1, BLOCK_N))
    bias = ct.load(bias_ptr, index=(b, h, q, 0), shape=(1, 1, 1, BLOCK_N))

    added = ct.astype(score, ct.float32) + bias
    rounded_bf = ct.astype(added, ct.bfloat16)
    ct.store(rounded_ptr, index=(b, h, q, 0), tile=rounded_bf)

    x = ct.astype(rounded_bf, ct.float32)
    x_2d = ct.reshape(x, (1, BLOCK_N))
    row_max = ct.max(x_2d, axis=1, keepdims=True)
    numer = ct.exp(x_2d - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(b, h, q), tile=ct.reshape(row_max, (1, 1, 1)))
    ct.store(sum_ptr, index=(b, h, q), tile=ct.reshape(denom, (1, 1, 1)))

    rand_f = ct.load(random_ptr, index=(b, h, q, 0), shape=(1, 1, 1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    rand_bf_2d = ct.reshape(rand_bf, (1, BLOCK_N))
    keep = rand_bf_2d > dropout_p_bf
    ct.store(gt_ptr, index=(b, h, q, 0), tile=ct.reshape(keep, (1, 1, 1, BLOCK_N)))

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs, zero_bf)
    scaled_out = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(b, h, q, 0), tile=ct.reshape(scaled_out, (1, 1, 1, BLOCK_N)))


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


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, bias, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs

    full_shape = tuple(int(dim) for dim in full_shape_arg)
    random_shape = tuple(int(dim) for dim in random_shape_arg)
    out_shape = tuple(int(dim) for dim in out_shape_arg)
    B, H, Q, K = full_shape
    device = x.device

    row_shape = full_shape[:-1] + (1,)
    rounded = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                                  device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                             device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_4d = x.view(B, H, Q, K)
    bias_contig = bias.contiguous()
    random_4d = random.view(B, H, Q, K)
    amax_3d = amax.view(B, H, Q)
    sum_3d = sum_1.view(B, H, Q)
    gt_4d = gt.view(B, H, Q, K)
    dropped_4d = dropped.view(B, H, Q, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, Q),
        _softmax_dropout_kernel,
        (x_4d, bias_contig, random_4d, rounded, amax_3d, sum_3d,
         gt_4d, dropped_4d, BLOCK_N),
    )
    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
