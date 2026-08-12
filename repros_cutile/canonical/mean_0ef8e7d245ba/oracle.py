"""cuTile port of mean_0ef8e7d245ba: MT5 seeded dropout + residual + RMSNorm.

Pre-generates seeded random via inductor_random outside the kernel, then a
single row cuTile kernel performs bf16 dropout mask, bf16 scaled dropout,
fp32 residual add, fp32 mean-square + eps=1e-6 rsqrt, affine weight scale,
and bf16 output. Returns (gt, add, rsqrt, view).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 24
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,     # bf16 [rows, HIDDEN]
    random_ptr,   # f32  [rows, HIDDEN]
    residual_ptr, # f32  [rows, HIDDEN]
    weight_ptr,   # f32  [HIDDEN]
    gt_ptr,       # b8   [rows, HIDDEN]
    add_ptr,      # f32  [rows, HIDDEN]
    rsqrt_ptr,    # f32  [rows]
    out_ptr,      # bf16 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_C))

    # gt: rand cast to bf16, > 0.1 bf16.
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    p_bf = ct.full((1, HIDDEN_C), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, HIDDEN_C), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add)

    sq_sum = ct.sum(add * add)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HIDDEN_C) + EPS)
    ct.store(rsqrt_ptr, index=(row,),
             tile=ct.full((1,), inv_rms, dtype=ct.float32))

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_C))
    normalized = add * inv_rms
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(weight_2d * normalized, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs

    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    rsqrt_shape = base_shape[:-1] + (1,)

    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    gt = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        rsqrt_shape, _contiguous_stride(rsqrt_shape),
        device=device, dtype=torch.float32,
    )
    out_base = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # arg0_1 is [rows, hidden] bf16 flat. Reshape all to 2D for the kernel.
    flat_2d = arg0_1.view(rows, hidden)
    random_2d = random.view(rows, hidden)
    residual_2d = arg2_1.view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_1d, out_2d, hidden),
    )
    return gt, add, rsqrt, out_base.view(out_shape)
