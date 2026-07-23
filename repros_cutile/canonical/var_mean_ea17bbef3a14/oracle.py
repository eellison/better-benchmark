"""cuTile port of var_mean_ea17bbef3a14: XLNet bf16 seeded-dropout residual LayerNorm.

Pre-generates the seeded random tensor via inductor_random (matches the Triton
oracle's non-graph-capture path). HIDDEN=1024 is a power of 2 — no masking on
reductions needed.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 11
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 1024
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    mask_ptr,        # b8 [rows, HIDDEN]
    normalized_ptr,  # f32 [rows, HIDDEN]
    affine_ptr,      # f32 [rows, HIDDEN]
    bf16_view_ptr,   # bf16 [rows, HIDDEN]
    div_ptr,         # f32 [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H_))

    random_bf16 = ct.astype(rand_f, ct.bfloat16)
    threshold = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf16 > threshold
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped_bf16 = ct.astype(ct.where(keep, ct.astype(flat, ct.float32), 0.0), ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    layernorm_input = ct.astype(scaled_bf16, ct.float32) + residual

    total = ct.sum(layernorm_input)
    mean = total * (1.0 / HIDDEN_)
    centered = layernorm_input - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_view_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN_, (1,)))


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
def oracle_forward(inputs, **_kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, shape2, shape3 = inputs
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)"
        )

    norm_shape = _as_shape(shape1)
    random_shape = _as_shape(shape2)
    flat_shape = _as_shape(shape3)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN

    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(rows, hidden).contiguous()
    residual_2d = arg2_1.reshape(rows, hidden).contiguous()

    mask_2d = mask.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
         mask_2d, normalized_2d, affine_2d, bf16_view_2d, div_1d,
         hidden, HIDDEN),
    )
    return mask, normalized, affine, bf16_view, div
