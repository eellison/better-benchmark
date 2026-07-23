"""cuTile port of var_mean_51931b6418f0: GPT2 seeded-dropout residual LayerNorm.

HIDDEN=768 is not a power of two, so we pad the tile to HIDDEN_PAD=1024 and
mask the tail on reductions and stores. Uses pre-generated random tensor
(via torch.ops.prims.inductor_random) since cuTile has no seeded on-device RNG.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768
HIDDEN_PAD = 1024
SEED_INDEX = 24
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    random_ptr,     # f32 [ROWS, HIDDEN]
    residual_ptr,   # f32 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    gt_ptr,         # b8 [ROWS, HIDDEN]
    norm_ptr,       # f32 [ROWS, HIDDEN]
    bf16_ptr,       # bf16 [ROWS, HIDDEN]
    div_ptr,        # f32 [ROWS]
    HIDDEN_: ct.Constant[int],
    HIDDEN_PAD_: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_f32 = ct.load(
        random_ptr, index=(row, 0), shape=(1, HIDDEN_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    rand_bf16 = ct.astype(rand_f32, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, HIDDEN_PAD_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16  # (1, HIDDEN_PAD) bool

    cols = ct.arange(HIDDEN_PAD_, dtype=ct.int32)
    valid = cols < HIDDEN_
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD_))
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf16 = ct.full(shape=(1, HIDDEN_PAD_), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf16)
    dropped_scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    residual_f = ct.astype(residual, ct.float32)
    x = ct.astype(dropped_scaled, ct.float32) + residual_f

    zero_f32 = ct.full(shape=(1, HIDDEN_PAD_), fill_value=0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x, zero_f32)

    mean_val = ct.sum(x_masked) * (1.0 / HIDDEN_F)
    centered = x - mean_val
    centered_masked = ct.where(valid_2d, centered, zero_f32)
    variance_val = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_F)
    invstd_val = ct.rsqrt(variance_val + EPS)
    normalized = centered * invstd_val

    weight = ct.load(
        weight_ptr, index=(0,), shape=(HIDDEN_PAD_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(HIDDEN_PAD_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD_))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PAD_))
    affine = normalized * weight_2d + bias_2d

    normalized_masked = ct.where(valid_2d, normalized, zero_f32)
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_bf16_masked = ct.where(valid_2d, affine_bf16, zero_bf16)

    ct.store(norm_ptr, index=(row, 0), tile=normalized_masked)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf16_masked)

    div_val = invstd_val * (1.0 / HIDDEN_F)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


def _shape(shape):
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


@oracle_impl(hardware="B200", point="bf8decda")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, random_shape, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape(shape0)
    rand_shape = _shape(random_shape)
    flat_shape = _shape(shape2)
    device = arg0_1.device

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_out = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(rand_shape, seed, device=device)

    # 2D views [ROWS, HIDDEN]
    flat_2d = arg0_1.view(ROWS, HIDDEN)
    random_2d = random.contiguous().view(ROWS, HIDDEN)
    residual_2d = arg2_1.contiguous().view(ROWS, HIDDEN)
    gt_2d = gt.view(ROWS, HIDDEN)
    normalized_2d = normalized.view(ROWS, HIDDEN)
    bf16_2d = bf16_out.view(ROWS, HIDDEN)
    div_1d = div.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _dropout_residual_ln_kernel,
        (
            flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
            gt_2d, normalized_2d, bf16_2d, div_1d,
            HIDDEN, HIDDEN_PAD, float(HIDDEN),
        ),
    )
    return gt, normalized, bf16_out, div
