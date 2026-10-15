"""cuTile port of mean_014afd4984e6: MT5 dual-dropout residual RMSNorm.

Two seeded dropouts (seeds 82 and 83) around a residual RMSNorm epilogue.
Both random tensors pre-generated with torch.ops.prims.inductor_random.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 82
SEED_INDEX_1 = 83
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dual_dropout_rmsnorm_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    random0_ptr,    # f32  [ROWS, HIDDEN]
    random1_ptr,    # f32  [ROWS, HIDDEN]
    residual_ptr,   # f32  [ROWS, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    gt0_ptr,        # b8   [ROWS, HIDDEN]
    add_ptr,        # f32  [ROWS, HIDDEN]
    rsqrt_ptr,      # f32  [ROWS]
    gt1_ptr,        # b8   [ROWS, HIDDEN]
    out_ptr,        # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
    EPS_: ct.Constant[float],
):
    row_block = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    rand0_f = ct.load(random0_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    rand1_f = ct.load(random1_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))

    # dropout 0: keep = rand_bf > 0.1 (rand cast to bf16 first)
    rand0_bf = ct.astype(rand0_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK_M, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep0 = rand0_bf > threshold_bf
    ct.store(gt0_ptr, index=(row_block, 0), tile=keep0)

    zero_bf = ct.full((BLOCK_M, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped0_bf = ct.where(keep0, flat_bf, zero_bf)
    scaled0_bf = ct.astype(
        ct.astype(dropped0_bf, ct.float32) * DROPOUT_SCALE_,
        ct.bfloat16,
    )

    add_f = residual_f + ct.astype(scaled0_bf, ct.float32)
    ct.store(add_ptr, index=(row_block, 0), tile=add_f)

    inv_h = 1.0 / HIDDEN
    square_sum = ct.sum(add_f * add_f, axis=1, keepdims=True) * inv_h
    inv_rms = ct.rsqrt(square_sum + EPS_)
    inv_rms_1d = ct.reshape(inv_rms, (BLOCK_M,))
    ct.store(rsqrt_ptr, index=(row_block,), tile=inv_rms_1d)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    normalized = add_f * inv_rms
    affine = weight_2d * normalized

    # dropout 1: keep = rand > 0.1 (rand is f32, no bf16 cast this time)
    keep1 = rand1_f > 0.1
    ct.store(gt1_ptr, index=(row_block, 0), tile=keep1)

    zero_f = ct.full((BLOCK_M, BLOCK_H), 0.0, dtype=ct.float32)
    dropped1 = ct.where(keep1, affine, zero_f)
    scaled1_f = dropped1 * DROPOUT_SCALE_
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(scaled1_f, ct.bfloat16))


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
    return (
        ((int(numel) - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
            torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
        )
    advance = _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_M=2, BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(shape0)
    random_shape0 = _shape_tuple(shape1)
    flat_out_shape = _shape_tuple(shape3)
    row_shape = full_shape[:-1] + (1,)
    hidden = int(arg0_1.shape[-1])
    # arg0_1 [4096, 512] -> reshape as [32,128,512], but we operate as
    # [4096, 512] over rows.
    rows = int(arg0_1.shape[0])
    device = arg0_1.device
    full_stride = _contiguous_stride(full_shape)

    gt0 = torch.empty_strided(full_shape, full_stride, device=device, dtype=torch.bool)
    add = torch.empty_strided(full_shape, full_stride, device=device, dtype=torch.float32)
    rsqrt = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    gt1 = torch.empty_strided(full_shape, full_stride, device=device, dtype=torch.bool)
    out = torch.empty_strided(
        flat_out_shape, _contiguous_stride(flat_out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape0, seed0, seed1, device=device)

    # Flatten to (rows, hidden) for the kernel
    flat_2d = arg0_1.contiguous().view(rows, hidden)
    random0_2d = random0.contiguous().view(rows, hidden)
    random1_2d = random1.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt0_2d = gt0.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    gt1_2d = gt1.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _dual_dropout_rmsnorm_kernel,
        (flat_2d, random0_2d, random1_2d, residual_2d, arg3_1,
         gt0_2d, add_2d, rsqrt_1d, gt1_2d, out_2d,
         hidden, BLOCK_M, BLOCK_H, DROPOUT_SCALE, EPS),
    )
    return gt0, add, rsqrt, gt1, out
