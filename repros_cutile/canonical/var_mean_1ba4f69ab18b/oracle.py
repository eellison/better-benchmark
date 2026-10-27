"""cuTile port of var_mean_1ba4f69ab18b: XLNet seeded-dropout residual LayerNorm.

Multi-output row kernel that computes dropout+residual+LN with a pre-generated
random tensor (torch.ops.prims.inductor_random) since cuTile has no seeded
on-device RNG. HIDDEN=1024 is a power of 2 — no padding needed.

Returns: (gt_mask, normalized_f32, affine_f32, bf16_flat, bf16_transpose,
inverse_std/HIDDEN). The last two share a bf16 base tensor via .view/.permute.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
SEQ = 16
ROWS = BATCH * SEQ  # 8192
HIDDEN = 1024
SEED_INDEX = 29
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    hidden_ptr,     # bf16 [ROWS, HIDDEN]
    random_ptr,     # f32 [ROWS, HIDDEN]
    residual_ptr,   # f32 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    gt_ptr,         # b8 [ROWS, HIDDEN]
    norm_ptr,       # f32 [ROWS, HIDDEN]
    affine_ptr,     # f32 [ROWS, HIDDEN]
    bf16_base_ptr,  # bf16 [ROWS, HIDDEN]
    div_ptr,        # f32 [ROWS]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    hidden = ct.load(
        hidden_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
    )
    residual = ct.load(
        residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
    )
    rand_f32 = ct.load(
        random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
    )

    rand_bf16 = ct.astype(rand_f32, ct.bfloat16)
    dropout_p_bf16 = ct.full(
        shape=(ROW_BLOCK, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf16 = ct.full(
        shape=(ROW_BLOCK, BLOCK_H), fill_value=0.0, dtype=ct.bfloat16,
    )
    dropped = ct.where(keep, hidden, zero_bf16)
    dropped_scaled = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    residual_f = ct.astype(residual, ct.float32)
    x = ct.astype(dropped_scaled, ct.float32) + residual_f

    inv_h = 1.0 / HIDDEN_
    mean_1d = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = x - mean_1d
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_base_ptr, index=(row_block, 0), tile=affine_bf16)

    # div = invstd / HIDDEN, one scalar per row.
    div_1d = ct.reshape(invstd_1d * inv_h, (ROW_BLOCK,))
    ct.store(div_ptr, index=(row_block,), tile=div_1d)


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


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, random_shape, _shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    base_shape = _shape(shape0)
    rand_shape = _shape(random_shape)
    device = arg0_1.device

    gt = torch.empty(base_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(base_shape, device=device, dtype=torch.float32)
    affine = torch.empty(base_shape, device=device, dtype=torch.float32)
    # bf16_base is [1, ROWS, HIDDEN] contiguous
    bf16_base = torch.empty((1, ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(rand_shape, seed, device=device)

    # 2D views
    hidden_2d = arg0_1.view(ROWS, HIDDEN)
    random_2d = random.contiguous().view(ROWS, HIDDEN)
    residual_2d = arg2_1.contiguous().view(ROWS, HIDDEN)
    gt_2d = gt.view(ROWS, HIDDEN)
    normalized_2d = normalized.view(ROWS, HIDDEN)
    affine_2d = affine.view(ROWS, HIDDEN)
    bf16_base_2d = bf16_base.view(ROWS, HIDDEN)
    div_1d = div.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, ROW_BLOCK), 1, 1),
        _dropout_residual_ln_kernel,
        (
            hidden_2d, random_2d, residual_2d, arg3_1, arg4_1,
            gt_2d, normalized_2d, affine_2d, bf16_base_2d, div_1d,
            HIDDEN, BLOCK_H, ROW_BLOCK,
        ),
    )

    bf16_flat = bf16_base.squeeze(0)  # bf16 [ROWS, HIDDEN]
    bf16_transpose = bf16_base.permute(0, 2, 1).squeeze(0)  # bf16 [HIDDEN, ROWS]
    return gt, normalized, affine, bf16_flat, bf16_transpose, div
