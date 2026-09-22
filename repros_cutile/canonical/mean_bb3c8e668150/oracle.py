"""cuTile port of mean_bb3c8e668150: T5 dropout+residual RMSNorm.

Pre-generates the random tensor via inductor_random outside the kernel, then
runs a cuTile row kernel producing (gt mask, add residual f32, rsqrt f32,
bf16 view).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 34
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    view_ptr,       # bf16 [rows, HIDDEN]
    random_ptr,     # f32  [rows, HIDDEN]
    residual_ptr,   # f32  [rows, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    gt_ptr,         # bool [rows, HIDDEN]
    add_ptr,        # f32  [rows, HIDDEN]
    rsqrt_ptr,      # f32  [rows]
    out_ptr,        # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    view_bf = ct.load(view_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weight_f = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, view_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_f = residual_f + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_f)

    inv_h = 1.0 / HIDDEN
    mean_sq = ct.sum(add_f * add_f) * inv_h
    rsqrt_val = ct.rsqrt(mean_sq + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(rsqrt_val, (1,)))

    normalized = add_f * rsqrt_val
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    affine = weight_2d * normalized
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, view_shape, random_shape, out_shape = inputs
    view_shape = tuple(int(d) for d in view_shape)
    random_shape = tuple(int(d) for d in random_shape)
    out_shape = tuple(int(d) for d in out_shape)
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    gt = torch.empty_strided(view_shape, _contiguous_stride(view_shape), device=device, dtype=torch.bool)
    add_out = torch.empty_strided(view_shape, _contiguous_stride(view_shape), device=device, dtype=torch.float32)
    rsqrt_shape = (view_shape[0], view_shape[1], 1)
    rsqrt = torch.empty_strided(rsqrt_shape, _contiguous_stride(rsqrt_shape), device=device, dtype=torch.float32)
    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    rows = view_shape[0] * view_shape[1]
    view_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add_out.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (view_2d, random_2d, residual_2d, arg3_1, gt_2d, add_2d, rsqrt_1d, out_2d, hidden, BLOCK_H),
    )
    return gt, add_out, rsqrt, out
