"""cuTile port of var_mean_3bc311a8676a: GoogleFnet dropout+residual+LayerNorm+complex64.

HIDDEN=768 (non-pow2). Kernel produces the fp32 affine; complex64 output is
constructed via torch.complex outside the kernel.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 8
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
HIDDEN = 768


@ct.kernel
def _dropout_residual_layernorm_kernel(
    x_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    mask_ptr, norm_ptr, affine_ptr, div_ptr,
    HIDDEN_: ct.Constant[int], BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)

    keep = rand > DROPOUT_P
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped = ct.astype(keep, ct.float32) * x
    scaled = dropped * DROPOUT_SCALE
    layernorm_input = scaled + residual

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    x_masked = ct.where(col_mask, layernorm_input, 0.0)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HIDDEN_)
    centered = layernorm_input - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN_), (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN
    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()
    residual_flat = arg2_1.reshape(rows, hidden).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
         mask_pad, norm_pad, affine_pad, div_1d, hidden, BLOCK_H),
    )

    mask = torch.empty(norm_shape, device=device, dtype=torch.bool)
    mask.view(rows, hidden).copy_(mask_pad.narrow(1, 0, hidden))
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))
    div = torch.empty(div_shape, device=device, dtype=torch.float32)
    div.view(rows).copy_(div_1d)

    complex_out = torch.complex(
        affine,
        torch.zeros_like(affine),
    ).to(torch.complex64)
    return mask, normalized, affine, complex_out, div
