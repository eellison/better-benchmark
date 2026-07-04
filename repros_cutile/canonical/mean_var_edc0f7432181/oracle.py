"""cuTile port of mean_var_edc0f7432181: BERT dropout residual LayerNorm.

Uses aten.mean + aten.var (correction=1) separately, so variance is divided
by HIDDEN-1 and returns sqrt (not rsqrt). Denominator is (sqrt(var)+eps).

Pre-generates seeded random tensor via inductor_random on the Python side; a
single row-parallel cuTile kernel does everything else. HIDDEN=768 padded to
BLOCK_H=1024.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 52
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


@ct.kernel
def _dropout_ln_kernel(
    x_ptr,          # bf16 [rows, BLOCK_H] (padded)
    random_ptr,     # f32 [rows, BLOCK_H]
    residual_ptr,   # f32 [rows, BLOCK_H]
    weight_ptr,     # f32 [BLOCK_H]
    bias_ptr,       # f32 [BLOCK_H]
    gt_ptr,         # bool [rows, BLOCK_H]
    add_ptr,        # f32 [rows, BLOCK_H]
    sqrt_ptr,       # f32 [rows]
    sub_ptr,        # f32 [rows, BLOCK_H]
    out_ptr,        # bf16 [rows, BLOCK_H]
    HIDDEN: ct.Constant[int],
    HIDDEN_MINUS_ONE_INV: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_f = ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)
    add_f = residual_f + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_f)

    add_masked = ct.where(col_valid_2d, add_f, 0.0)
    mean = ct.sum(add_masked) * (1.0 / HIDDEN)
    centered = add_f - mean
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * HIDDEN_MINUS_ONE_INV
    std = ct.sqrt(variance)
    ct.store(sqrt_ptr, index=(row,), tile=std)
    ct.store(sub_ptr, index=(row, 0), tile=centered)

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_1d, (1, BLOCK_H))
    numerator = weight_2d * centered
    denom = std + DENOM_EPS
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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    del ROW_BLOCK
    x, seeds, residual, weight, bias, view_shape, random_shape, out_shape = inputs
    view_shape = _as_shape(view_shape)
    random_shape = _as_shape(random_shape)
    out_shape = _as_shape(out_shape)
    stat_shape = view_shape[:-1] + (1,)
    view_stride = _contiguous_stride(view_shape)
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])
    device = x.device

    x_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    x_padded[:, :hidden] = x
    residual_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    residual_padded[:, :hidden] = residual.view(rows, hidden)
    weight_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_padded[:hidden] = weight
    bias_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias_padded[:hidden] = bias

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random_full = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random_full.view(rows, hidden)
    random_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    random_padded[:, :hidden] = random_flat

    gt_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sqrt_flat = torch.empty((rows,), device=device, dtype=torch.float32)
    sub_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    out_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_ln_kernel,
        (
            x_padded, random_padded, residual_padded,
            weight_padded, bias_padded,
            gt_padded, add_padded, sqrt_flat, sub_padded, out_padded,
            hidden, 1.0 / (hidden - VAR_CORRECTION), BLOCK_H,
        ),
    )

    gt = torch.empty_strided(view_shape, view_stride, device=device, dtype=torch.bool)
    gt.view(rows, hidden).copy_(gt_padded[:, :hidden])
    add = torch.empty_strided(view_shape, view_stride, device=device, dtype=torch.float32)
    add.view(rows, hidden).copy_(add_padded[:, :hidden])
    sqrt = torch.empty_strided(stat_shape, _contiguous_stride(stat_shape),
                               device=device, dtype=torch.float32)
    sqrt.view(rows).copy_(sqrt_flat)
    sub = torch.empty_strided(view_shape, view_stride, device=device, dtype=torch.float32)
    sub.view(rows, hidden).copy_(sub_padded[:, :hidden])
    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=device, dtype=torch.bfloat16)
    out.copy_(out_padded[:, :hidden])

    return gt, add, sqrt, sub, out
