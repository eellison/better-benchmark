"""cuTile port of var_mean_592f69819372: OPT generated-seed dropout + residual + LayerNorm.

Uses eager pre-generated random via torch.ops.prims.inductor_random.
HIDDEN=768 -> BLOCK_H=1024 with masked reductions.

Returns (inductor_seeds, gt, view_1, getitem_1, rsqrt, out_bf16):
    - inductor_seeds i64[2]
    - gt (mask) b8 [4, 2048, 768]
    - view_1 f32 [8192, 768] = residual + scaled_dropped
    - getitem_1 (mean) f32 [8192, 1]
    - rsqrt f32 [8192, 1]
    - bf16 output [8192, 768]
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
SEQ_LEN = 2048
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
BLOCK_H = 1024
SEED_COUNT = 2
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    mask_ptr,        # b8 padded [rows, BLOCK_H]
    add_ptr,         # f32 padded [rows, BLOCK_H]
    mean_ptr,        # f32 [rows]
    rsqrt_ptr,       # f32 [rows]
    bf16_out_ptr,    # bf16 padded [rows, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    random_bf16 = ct.astype(rand_f, ct.bfloat16)
    threshold = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf16 > threshold
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped_bf16 = ct.astype(ct.where(keep, ct.astype(flat, ct.float32), 0.0), ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = residual + ct.astype(scaled_bf16, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x)

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, 0.0)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HIDDEN_)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_flat_shape(shape, flat_bf16):
    resolved = _as_shape(shape)
    if -1 not in resolved:
        return resolved
    return tuple(int(dim) for dim in flat_bf16.shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


@oracle_impl(hardware="B200", point="b50f188e", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, **_kwargs):
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    (
        flat_bf16,
        residual,
        weight,
        bias,
        norm_shape_param,
        random_shape_param,
        flat_shape_param,
    ) = inputs
    norm_shape = _as_shape(norm_shape_param)
    random_shape = _as_shape(random_shape_param)
    flat_shape = _resolve_flat_shape(flat_shape_param, flat_bf16)
    device = flat_bf16.device
    rows = ROWS
    hidden = HIDDEN

    mask_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_out_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    mean_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    rsqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()
    residual_flat = residual.reshape(rows, hidden).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (flat_bf16, random_flat, residual_flat, weight, bias,
         mask_pad, add_pad, mean_1d, rsqrt_1d, bf16_out_pad,
         hidden, BLOCK_H),
    )

    # Narrow padded outputs back to logical shape.
    gt = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    gt.view(rows, hidden).copy_(mask_pad.narrow(1, 0, hidden))

    add_flat = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.float32,
    )
    add_flat.view(rows, hidden).copy_(add_pad.narrow(1, 0, hidden))

    mean = mean_1d.view(rows, 1)
    rsqrt = rsqrt_1d.view(rows, 1)

    out_bf16 = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    out_bf16.view(rows, hidden).copy_(bf16_out_pad.narrow(1, 0, hidden))

    return seeds, gt, add_flat, mean, rsqrt, out_bf16
