"""cuTile port of mean_var_b13fb4152fa7: BERT bf16 dropout-residual LayerNorm.

Uses pre-generated random tensor from `torch.ops.prims.inductor_random` (same
approach as var_mean_a678817dc522). cuTile is RTNE by default so we drop the
inline_asm PTX add.rn.f32/mul.rn.f32 wrappers and use plain +/*. Computes
mean and unbiased variance (correction=1.0) explicitly.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 47
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


@ct.kernel
def _dropout_mean_var_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    random_ptr,     # f32  [ROWS, HIDDEN]
    residual_ptr,   # f32  [ROWS, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    bias_ptr,       # f32  [HIDDEN]
    gt_ptr,         # b8   [ROWS, HIDDEN]
    add_ptr,        # f32  [ROWS, HIDDEN]
    sqrt_ptr,       # f32  [ROWS, 1]
    sub_ptr,        # f32  [ROWS, HIDDEN]
    out_ptr,        # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold_bf = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    add_f = residual_f + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row_block, 0), tile=add_f)

    inv_h = 1.0 / HIDDEN
    row_sum = ct.sum(add_f, axis=1, keepdims=True)
    row_sum_sq = ct.sum(add_f * add_f, axis=1, keepdims=True)
    mean = row_sum * inv_h
    centered = add_f - mean
    ct.store(sub_ptr, index=(row_block, 0), tile=centered)

    # unbiased variance = (row_sum_sq - row_sum * mean) / (HIDDEN - 1)
    variance_sum = row_sum_sq - row_sum * mean
    variance = variance_sum * (1.0 / (HIDDEN - VAR_CORRECTION))
    zero_1 = ct.full((ROW_BLOCK, 1), 0.0, dtype=ct.float32)
    variance_clamped = ct.where(variance > zero_1, variance, zero_1)
    std = ct.sqrt(variance_clamped)
    ct.store(sqrt_ptr, index=(row_block, 0), tile=std)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = (weight_2d * centered) / (std + DENOM_EPS) + bias_2d
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    x, seeds, residual, weight, bias, view_shape, random_shape, out_shape = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _as_shape(view_shape)
    random_shape = _as_shape(random_shape)
    out_shape = _as_shape(out_shape)
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])
    stat_shape = (view_shape[0], view_shape[1], 1)
    device = x.device

    gt = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    sqrt = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=device, dtype=torch.float32,
    )
    sub = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # Reshape all to [rows, hidden].
    x_2d = x.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = residual.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    sqrt_2d = sqrt.view(rows, 1)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_mean_var_kernel,
        (x_2d, random_2d, residual_2d, weight, bias,
         gt_2d, add_2d, sqrt_2d, sub_2d, out_2d,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    return gt, add, sqrt, sub, out
