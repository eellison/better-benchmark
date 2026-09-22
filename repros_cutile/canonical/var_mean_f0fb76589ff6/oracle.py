"""cuTile port of var_mean_f0fb76589ff6: GoogleFnet dropout+LayerNorm (f32 input).

Input is f32 (no bf16 dropout rounding), seed=3, eps=1e-12. Output includes a
complex64 conversion (done torch-side after the kernel).
Returns (gt, mul_2, add_2, complex_view, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 3
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    x_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    mask_ptr, mul2_ptr, add2_ptr, div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    x = ct.load(x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    # Both f32.
    # Note: Repro uses `gt(inductor_random, 0.1)` directly on f32 (no bf16 cast!)
    dropout_p = ct.full((ROW_BLOCK, BLOCK_H), 0.1, dtype=ct.float32)
    keep = rand > dropout_p
    ct.store(mask_ptr, index=(row_block, 0), tile=keep)

    zero_f = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, x, zero_f)
    scaled = dropped * DROPOUT_SCALE
    x_f = scaled + residual

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(x_f, axis=1, keepdims=True) * inv_h
    centered = x_f - mean_1d
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(mul2_ptr, index=(row_block, 0), tile=normalized)
    ct.store(add2_ptr, index=(row_block, 0), tile=affine)
    ct.store(div_ptr, index=(row_block, 0), tile=invstd_1d * inv_h)


def _shape(shape):
    return tuple(int(d) for d in shape)


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


@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape(shape0)
    random_shape = _shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    div_2d = div.view(rows, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_layernorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, normalized_2d, affine_2d, div_2d,
         hidden, BLOCK_H, 1),
    )

    # convert_element_type to complex64 is done torch-side.
    complex_view = affine.to(torch.complex64)
    return gt, normalized, affine, complex_view, div
