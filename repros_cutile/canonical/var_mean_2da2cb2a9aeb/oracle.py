"""cuTile port of var_mean_2da2cb2a9aeb: DebertaV2/Bert-family dropout + LayerNorm row kernel.

Uses eager pre-generated random via torch.ops.prims.inductor_random (matches the
Triton oracle's non-graph-capture path). Multiple HIDDEN sizes are handled via
padded BLOCK_H with masked reductions.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 20
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


@ct.kernel
def _dropout_residual_layernorm_kernel(
    addmm_ptr,       # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    gt_ptr,          # b8 padded [rows, BLOCK_H]
    norm_ptr,        # f32 padded [rows, BLOCK_H]
    affine_ptr,      # f32 padded [rows, BLOCK_H]
    bf16_ptr,        # bf16 padded [rows, BLOCK_H]
    div_ptr,         # f32 [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    addmm = ct.load(
        addmm_ptr, index=(row, 0), shape=(1, BLOCK_H_),
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
    threshold_bf16 = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf16 > threshold_bf16
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf16 = ct.astype(ct.where(keep, ct.astype(addmm, ct.float32), 0.0), ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled_bf16, ct.float32) + residual

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
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN_, (1,)))


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
    advance = (numel + 131071) // 131072
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


# 55aa5fd0: HIDDEN=1536, BLOCK_H=2048
@oracle_impl(hardware="B200", point="55aa5fd0", BLOCK_H=2048, ROW_BLOCK=1)
# 243d7832: HIDDEN=768, BLOCK_H=1024
@oracle_impl(hardware="B200", point="243d7832", BLOCK_H=1024, ROW_BLOCK=1)
# d9ecc504: HIDDEN=256, BLOCK_H=256
@oracle_impl(hardware="B200", point="d9ecc504", BLOCK_H=256, ROW_BLOCK=2)
def oracle_forward(inputs, *, BLOCK_H: int, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape2)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    affine = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()
    residual_flat = arg2_1.reshape(rows, hidden).contiguous()

    # Padded scratch buffers to handle BLOCK_H > HIDDEN cases.
    if BLOCK_H > hidden:
        gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
        normalized_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
        affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
        bf16_view_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    else:
        gt_pad = gt.view(rows, hidden)
        normalized_pad = normalized.view(rows, hidden)
        affine_pad = affine.view(rows, hidden)
        bf16_view_pad = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
         gt_pad, normalized_pad, affine_pad, bf16_view_pad, div_1d,
         hidden, BLOCK_H),
    )

    if BLOCK_H > hidden:
        gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
        normalized.view(rows, hidden).copy_(normalized_pad.narrow(1, 0, hidden))
        affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))
        bf16_view.view(rows, hidden).copy_(bf16_view_pad.narrow(1, 0, hidden))

    return (gt, normalized, affine, bf16_view, div)
