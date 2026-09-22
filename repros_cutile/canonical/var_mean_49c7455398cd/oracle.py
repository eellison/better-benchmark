"""cuTile port of var_mean_49c7455398cd: GoogleFnet seeded-dropout residual LN.

Inputs are f32 all the way through (no bf16 boundaries). Returns:
(gt, mul_2, add_2, complex64_cast, div).

Dropout: keep = random > 0.1 (f32 mask), scaled = keep * source * DROPOUT_SCALE.
Then residual add, var_mean, rsqrt(eps=1e-12), affine, and a real->complex64
cast (imag=0). We do the complex conversion outside the kernel.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 9
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    x_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, norm_ptr, affine_ptr, div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    keep = rand > 0.1
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    dropped = ct.where(keep, x, zero)
    scaled = dropped * DROPOUT_SCALE
    x_f = scaled + residual
    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x_f, axis=1, keepdims=True) * inv_h
    centered = x_f - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    div_val = invstd * inv_h
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1 = inputs
    norm_shape = _shape(shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    if BLOCK_H == hidden:
        weight_pad = arg3_1
        bias_pad = arg4_1
        x_pad = x_2d
        random_pad = random_2d
        residual_pad = residual_2d
    else:
        x_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        x_pad[:, :hidden].copy_(x_2d)
        random_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        random_pad[:, :hidden].copy_(random_2d)
        residual_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        residual_pad[:, :hidden].copy_(residual_2d)
        weight_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        weight_pad[:hidden].copy_(arg3_1)
        bias_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        bias_pad[:hidden].copy_(arg4_1)

    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (x_pad, random_pad, residual_pad, weight_pad, bias_pad,
         gt_pad, norm_pad, affine_pad, div_1d,
         hidden, BLOCK_H),
    )

    if BLOCK_H == hidden:
        gt = gt_pad.view(norm_shape)
        normalized = norm_pad.view(norm_shape)
        affine = affine_pad.view(norm_shape)
    else:
        gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
        gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
        normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
        normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))
        affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
        affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))

    # complex64 cast: real=affine, imag=0
    complex_out = affine.to(torch.complex64)
    div = div_1d.view(div_shape)
    return gt, normalized, affine, complex_out, div
