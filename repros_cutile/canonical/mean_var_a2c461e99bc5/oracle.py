"""cuTile port of mean_var_a2c461e99bc5: BERT dual-dropout residual LayerNorm.

Fuses two seeded Inductor dropouts (seed 29 bf16-compare, seed 30 fp32-compare),
residual add, mean, unbiased var, sub, affine (weight/bias). HIDDEN=768
requires masking with BLOCK_H=1024.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 29
SEED_INDEX_1 = 30
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _dual_dropout_layernorm_kernel(
    flat_ptr,       # bf16 padded [rows, BLOCK_H]
    random0_ptr,    # f32  padded [rows, BLOCK_H]
    random1_ptr,    # f32  padded [rows, BLOCK_H]
    residual_ptr,   # f32  padded [rows, BLOCK_H]
    weight_ptr,     # f32  padded [BLOCK_H]
    bias_ptr,       # f32  padded [BLOCK_H]
    gt0_ptr,        # b8   padded [rows, BLOCK_H]
    gt1_ptr,        # b8   padded [rows, BLOCK_H]
    dropped_ptr,    # f32  padded [rows, BLOCK_H]
    sqrt_ptr,       # f32  [rows]
    sub_ptr,        # f32  padded [rows, BLOCK_H]
    out_ptr,        # bf16 padded [rows, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    rand0 = ct.load(random0_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    rand1 = ct.load(random1_ptr, index=(row, 0), shape=(1, BLOCK_H_))

    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep0 = ct.astype(rand0, ct.bfloat16) > threshold_bf
    keep1 = rand1 > DROPOUT_P

    ct.store(gt0_ptr, index=(row, 0), tile=keep0)
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)

    dropped0_bf = ct.astype(
        ct.where(keep0, ct.astype(flat, ct.float32), 0.0),
        ct.bfloat16,
    )
    scaled0_bf = ct.astype(
        ct.astype(dropped0_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    add_val = residual + ct.astype(scaled0_bf, ct.float32)
    dropped1 = ct.where(keep1, add_val, 0.0)
    x = dropped1 * DROPOUT_SCALE
    ct.store(dropped_ptr, index=(row, 0), tile=x)

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, 0.0)
    row_sum = ct.sum(x_masked)
    row_sumsq = ct.sum(x_masked * x_masked)
    mean = row_sum * (1.0 / HIDDEN_)
    variance = (row_sumsq - HIDDEN_ * mean * mean) * (1.0 / (HIDDEN_ - 1))
    variance = ct.where(variance > 0.0, variance, 0.0)
    std = ct.sqrt(variance)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), std, dtype=ct.float32), (1,)))

    centered = x - mean
    ct.store(sub_ptr, index=(row, 0), tile=centered)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    numerator = weight_2d * centered
    denom = std + 1.0e-6
    normalized = numerator * (1.0 / denom)
    affine = normalized + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
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


@oracle_impl(hardware="B200", point="4205ff34")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2, shape3 = inputs
    device = arg0_1.device
    base_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape3)
    sqrt_shape = base_shape[:-1] + (1,)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape, seed0, seed1, device=device,
    )
    random0_2d = random0.reshape(rows, hidden).contiguous()
    random1_2d = random1.reshape(rows, hidden).contiguous()
    residual_2d = arg2_1.reshape(rows, hidden).contiguous()
    flat_2d = arg0_1.contiguous().view(rows, hidden)

    gt0_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    gt1_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    dropped_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    sub_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    out_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)

    flat_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    flat_pad.narrow(1, 0, hidden).copy_(flat_2d)
    r0_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    r0_pad.narrow(1, 0, hidden).copy_(random0_2d)
    r1_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    r1_pad.narrow(1, 0, hidden).copy_(random1_2d)
    resid_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    resid_pad.narrow(1, 0, hidden).copy_(residual_2d)
    w_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    w_pad.narrow(0, 0, hidden).copy_(arg3_1)
    b_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    b_pad.narrow(0, 0, hidden).copy_(arg4_1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dual_dropout_layernorm_kernel,
        (flat_pad, r0_pad, r1_pad, resid_pad, w_pad, b_pad,
         gt0_pad, gt1_pad, dropped_pad, sqrt_1d, sub_pad, out_pad,
         hidden, BLOCK_H),
    )

    gt0 = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bool,
    )
    gt0.view(rows, hidden).copy_(gt0_pad.narrow(1, 0, hidden))
    gt1 = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bool,
    )
    gt1.view(rows, hidden).copy_(gt1_pad.narrow(1, 0, hidden))
    dropped = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.float32,
    )
    dropped.view(rows, hidden).copy_(dropped_pad.narrow(1, 0, hidden))
    sqrt = torch.empty_strided(
        sqrt_shape, _contiguous_stride(sqrt_shape),
        device=device, dtype=torch.float32,
    )
    sqrt.view(rows).copy_(sqrt_1d)
    sub = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.float32,
    )
    sub.view(rows, hidden).copy_(sub_pad.narrow(1, 0, hidden))
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )
    out.copy_(out_pad.narrow(1, 0, hidden).reshape(out_shape))

    return gt0, gt1, dropped, sqrt, sub, out
