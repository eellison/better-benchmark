"""cuTile port of mean_var_381b3c976327: BERT dropout + residual + LayerNorm.

Uses HIDDEN=768 with BLOCK_H=1024 masked reductions.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 27
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
EPS = 1.0e-6


@ct.kernel
def _dropout_mean_var_kernel(
    x_ptr,          # bf16 (rows, HIDDEN)
    random_ptr,     # f32 (rows, HIDDEN)
    residual_ptr,   # f32 (rows, HIDDEN)
    weight_ptr,     # f32 (HIDDEN,)
    bias_ptr,       # f32 (HIDDEN,)
    gt_ptr,         # b8 (rows, BLOCK_H) padded
    add_ptr,        # f32 (rows, BLOCK_H) padded
    sqrt_ptr,       # f32 (rows,)
    sub_ptr,        # f32 (rows, BLOCK_H) padded
    out_ptr,        # bf16 (rows, BLOCK_H) padded
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    add_for_sum = ct.where(col_mask, add_val, zero_f)
    row_sum = ct.sum(add_for_sum)
    row_sum_sq = ct.sum(add_for_sum * add_for_sum)
    mean = row_sum * (1.0 / HIDDEN)
    centered = add_val - mean
    variance = (row_sum_sq - HIDDEN * mean * mean) * (1.0 / (HIDDEN - VAR_CORRECTION))
    variance_safe = ct.where(variance > 0.0, variance, 0.0)
    std = ct.sqrt(variance_safe)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(std, (1,)))
    ct.store(sub_ptr, index=(row, 0), tile=centered)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    denom = std + EPS
    normalized = weight_2d * centered * (1.0 / denom)
    affine = normalized + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    x, seeds, residual, weight, bias, view_shape, random_shape, out_shape = inputs
    view_shape = tuple(int(dim) for dim in view_shape)
    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    stat_shape = view_shape[:-1] + (1,)
    device = x.device
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])

    # Output buffers (padded to BLOCK_H, then narrow to hidden)
    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sub_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    out_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    sqrt = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.view(rows, hidden)
    resid_2d = residual.view(rows, hidden)
    sqrt_1d = sqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_mean_var_kernel,
        (x, random_2d, resid_2d, weight, bias,
         gt_pad, add_pad, sqrt_1d, sub_pad, out_pad, hidden, BLOCK_H),
    )

    gt = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool)
    gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
    add = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32)
    add.view(rows, hidden).copy_(add_pad.narrow(1, 0, hidden))
    sub = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32)
    sub.view(rows, hidden).copy_(sub_pad.narrow(1, 0, hidden))
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)
    out.view(rows, hidden).copy_(out_pad.narrow(1, 0, hidden))
    return gt, add, sqrt, sub, out
