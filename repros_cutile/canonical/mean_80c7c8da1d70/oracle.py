"""cuTile port of mean_80c7c8da1d70: MT5 dual-dropout residual RMSNorm.

Two seeded random tensors pre-generated outside the kernel. Single cuTile
row kernel does: dropout0 -> residual add -> mean(square) rsqrt -> weight mul ->
dropout1 -> scale -> bf16 store.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 32
SEED_INDEX_1 = 33
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dual_dropout_rmsnorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random0_ptr,     # f32  [rows, HIDDEN]
    random1_ptr,     # f32  [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    gt0_ptr,         # b8  [rows, HIDDEN]
    add_ptr,         # f32 [rows, HIDDEN]
    rsqrt_ptr,       # f32 [rows]
    gt1_ptr,         # b8  [rows, HIDDEN]
    mul5_ptr,        # f32 [rows, HIDDEN]
    out_ptr,         # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand0_f = ct.load(random0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand1_f = ct.load(random1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand0_bf = ct.astype(rand0_f, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep0 = rand0_bf > thresh_bf
    ct.store(gt0_ptr, index=(row, 0), tile=keep0)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped0 = ct.where(keep0, flat_bf, zero_bf)
    scaled0 = ct.astype(ct.astype(dropped0, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(scaled0, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add)

    inv_h = 1.0 / HIDDEN
    square_sum = ct.sum(add * add, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(square_sum * inv_h + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(inv_rms, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    normalized = add * inv_rms
    affine = weight_2d * normalized

    thresh_f = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.float32)
    keep1 = rand1_f > thresh_f
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    dropped1 = ct.where(keep1, affine, zero_f)
    mul5 = dropped1 * DROPOUT_SCALE
    ct.store(mul5_ptr, index=(row, 0), tile=mul5)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(mul5, ct.bfloat16))


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


def _random_advance(shape, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (int(numel) + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    return (((int(numel) - 1) // (block_size * grid * unroll) + 1) * curand4_engine_calls * 2)


def _inductor_random_pair(shape, seed0, seed1, *, device):
    advance = _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        r0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        r1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return r0, r1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, _shape2, shape3 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(shape0)  # [32,128,512]
    random_shape = _shape_tuple(shape1)  # [32,128,512]
    flat_out_shape = _shape_tuple(shape3)  # [4096,512]
    row_shape = full_shape[:-1] + (1,)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    device = arg0_1.device

    gt0 = torch.empty_strided(full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.bool)
    add = torch.empty_strided(full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.float32)
    rsqrt = torch.empty_strided(row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32)
    gt1 = torch.empty_strided(full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.bool)
    mul5 = torch.empty_strided(full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.float32)
    out = torch.empty_strided(flat_out_shape, _contiguous_stride(flat_out_shape), device=device, dtype=torch.bfloat16)

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
    random0, random1 = _inductor_random_pair(random_shape, seed0, seed1, device=device)

    x_2d = arg0_1.view(rows, hidden)
    random0_2d = random0.contiguous().view(rows, hidden)
    random1_2d = random1.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    weight_1d = arg3_1.view(hidden)
    gt0_2d = gt0.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    gt1_2d = gt1.view(rows, hidden)
    mul5_2d = mul5.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dual_dropout_rmsnorm_kernel,
        (x_2d, random0_2d, random1_2d, residual_2d, weight_1d,
         gt0_2d, add_2d, rsqrt_1d, gt1_2d, mul5_2d, out_2d,
         hidden, BLOCK_H),
    )
    return gt0, add, rsqrt, gt1, mul5, out
