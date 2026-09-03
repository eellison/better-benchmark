"""cuTile port of mean_var_ea8b03f394dc: BERT dropout + residual + dropout + LayerNorm.

Uses eager pre-generated random tensors via torch.ops.prims.inductor_random
(matches the Triton oracle's non-graph-capture path). HIDDEN=768 -> BLOCK_H=1024
with masked reductions. Raises NotImplementedError under CUDA graph capture
because seeded on-device RNG is not portable to cuTile.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 19
SEED_INDEX_1 = 20
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6
HIDDEN = 768


@ct.kernel
def _dropout_dropout_mean_var_kernel(
    x_ptr,           # bf16 [rows, HIDDEN]
    random0_ptr,     # f32  [rows, HIDDEN]
    random1_ptr,     # f32  [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    bias_ptr,        # f32  [HIDDEN]
    gt0_pad_ptr,     # b8   padded [rows, BLOCK_H]
    gt1_pad_ptr,     # b8   padded [rows, BLOCK_H]
    dropped_pad_ptr, # f32  padded [rows, BLOCK_H]
    sqrt_ptr,        # f32  [rows]
    sub_pad_ptr,     # f32  padded [rows, BLOCK_H]
    out_pad_ptr,     # bf16 padded [rows, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random0 = ct.load(
        random0_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random1 = ct.load(
        random1_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )

    threshold_bf16 = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    random0_bf16 = ct.astype(random0, ct.bfloat16)
    keep0 = random0_bf16 > threshold_bf16
    ct.store(gt0_pad_ptr, index=(row, 0), tile=keep0)

    first_dropped = ct.astype(ct.where(keep0, ct.astype(x, ct.float32), 0.0), ct.bfloat16)
    first_scaled = ct.astype(
        ct.astype(first_dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    add = residual + ct.astype(first_scaled, ct.float32)

    keep1 = random1 > DROPOUT_P
    ct.store(gt1_pad_ptr, index=(row, 0), tile=keep1)
    norm_input = ct.where(keep1, add, 0.0) * DROPOUT_SCALE
    ct.store(dropped_pad_ptr, index=(row, 0), tile=norm_input)

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    valid_input = ct.where(col_mask, norm_input, 0.0)
    row_sum = ct.sum(valid_input)
    row_sum_sq = ct.sum(valid_input * valid_input)
    mean = row_sum / HIDDEN_
    centered = norm_input - mean
    variance_sum = row_sum_sq - row_sum * mean
    variance = variance_sum / (HIDDEN_ - VAR_CORRECTION)
    variance_max = ct.where(variance > 0.0, variance, 0.0)
    std = ct.sqrt(variance_max)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(ct.full((1,), std, dtype=ct.float32), (1,)))
    ct.store(sub_pad_ptr, index=(row, 0), tile=centered)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = (weight_2d * centered) / (std + DENOM_EPS) + bias_2d
    ct.store(out_pad_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


def _inductor_random_advance(shape, *, device):
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


def _inductor_random_for_eager_check(shape, seed, *, device, calls_back):
    advance = _inductor_random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewind = advance * calls_back
    if offset >= rewind:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - rewind)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    del ROW_BLOCK
    (
        x,
        seeds,
        residual,
        weight,
        bias,
        view_shape,
        random0_shape,
        random1_shape,
        out_shape,
    ) = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _as_shape(view_shape)
    random0_shape = _as_shape(random0_shape)
    random1_shape = _as_shape(random1_shape)
    out_shape = _as_shape(out_shape)
    device = x.device
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])
    assert hidden == HIDDEN
    stat_shape = (view_shape[0], view_shape[1], 1)

    # Pre-generate the two random tensors (must precede kernel launch to match
    # Triton oracle's use of _inductor_random_for_eager_check).
    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
    random0 = _inductor_random_for_eager_check(
        random0_shape, seed0, device=device, calls_back=2,
    )
    random1 = _inductor_random_for_eager_check(
        random1_shape, seed1, device=device, calls_back=1,
    )
    random0_flat = random0.reshape(rows, hidden).contiguous()
    random1_flat = random1.reshape(rows, hidden).contiguous()
    residual_flat = residual.reshape(rows, hidden).contiguous()

    gt0_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    gt1_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    dropped_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sub_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    out_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    sqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_dropout_mean_var_kernel,
        (
            x, random0_flat, random1_flat, residual_flat, weight, bias,
            gt0_pad, gt1_pad, dropped_pad, sqrt_1d, sub_pad, out_pad,
            hidden, BLOCK_H,
        ),
    )

    gt0 = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    gt0.view(rows, hidden).copy_(gt0_pad.narrow(1, 0, hidden))

    gt1 = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    gt1.view(rows, hidden).copy_(gt1_pad.narrow(1, 0, hidden))

    dropped = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    dropped.view(rows, hidden).copy_(dropped_pad.narrow(1, 0, hidden))

    sqrt = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=device, dtype=torch.float32,
    )
    sqrt.view(rows).copy_(sqrt_1d)

    sub = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    sub.view(rows, hidden).copy_(sub_pad.narrow(1, 0, hidden))

    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )
    out.view(rows, hidden).copy_(out_pad.narrow(1, 0, hidden))

    return gt0, gt1, dropped, sqrt, sub, out
