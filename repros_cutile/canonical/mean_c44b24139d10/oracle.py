"""cuTile port of mean_c44b24139d10: MT5/T5 bf16 dropout-residual RMSNorm.

Uses pre-generated random tensor from `torch.ops.prims.inductor_random` (same
approach as var_mean_a678817dc522). cuTile is RTNE by default so we drop the
inline_asm PTX add.rn.f32/mul.rn.f32 wrappers and use plain +/*. Rounds bf16
via `ct.astype(..., ct.bfloat16)`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 56
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
RMS_EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    random_ptr,     # f32  [ROWS, HIDDEN]
    residual_ptr,   # f32  [ROWS, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    gt_ptr,         # b8   [ROWS, HIDDEN]
    add_ptr,        # f32  [ROWS, HIDDEN]
    rsqrt_ptr,      # f32  [ROWS, 1]
    out_ptr,        # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    src = ct.load(flat_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, src, zero_bf)
    dropped_scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    x_f = residual_f + ct.astype(dropped_scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row_block, 0), tile=x_f)

    # RMSNorm: mean(x^2) then rsqrt(mean_sq + eps).
    inv_h = 1.0 / HIDDEN
    square_sum = ct.sum(x_f * x_f, axis=1, keepdims=True)
    mean_square = square_sum * inv_h
    inv_rms = ct.rsqrt(mean_square + RMS_EPS)  # shape (ROW_BLOCK, 1)
    ct.store(rsqrt_ptr, index=(row_block, 0), tile=inv_rms)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    normed = x_f * inv_rms
    out = weight_2d * normed
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(out, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    rsqrt_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        rsqrt_shape, _contiguous_stride(rsqrt_shape),
        device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # Reshape all to [rows, hidden].
    src_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_2d = rsqrt.view(rows, 1)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (src_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_2d, out_2d,
         hidden, BLOCK_H, ROW_BLOCK),
    )
    return (gt, add, rsqrt, out)
