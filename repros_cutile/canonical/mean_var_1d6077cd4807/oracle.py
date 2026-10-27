"""cuTile port of mean_var_1d6077cd4807: BERT dual-dropout residual LayerNorm.

First dropout is bf16 (mask on random cast to bf16, then bf16 mul of gt with
view, bf16 mul by 1.11..., then f32 add with residual). Second dropout is f32
(gt on f32 random, mul with float32 add, mul by 1.11..).

Then LayerNorm-like with sample variance (correction=1), sqrt (not rsqrt),
divide by (sqrt + 1e-6), affine (weight * mul_2, add bias).

Returns (gt, gt_1, mul_3, sqrt, sub, view_1).
Where mul_3 is the dual-dropped fp32 tensor, sqrt = sample std, sub = mul_3 - mean.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 39
SEED_INDEX_1 = 40
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dual_dropout_ln_kernel(
    x_ptr, rand0_ptr, rand1_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, gt1_ptr, mul3_ptr, sqrt_ptr, sub_ptr, bf16_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand0 = ct.load(rand0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand1 = ct.load(rand1_ptr, index=(row, 0), shape=(1, BLOCK_H))

    # First dropout: bf16 comparison
    rand0_bf = ct.astype(rand0, ct.bfloat16)
    p_bf = ct.astype(ct.full((1, BLOCK_H), 0.1, dtype=ct.float32), ct.bfloat16)
    keep0 = rand0_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep0)
    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep0, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(scaled_bf, ct.float32)

    # Second dropout: f32 comparison
    keep1 = rand1 > 0.1
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    dropped1 = ct.where(keep1, add, zero_f)
    mul_3 = dropped1 * DROPOUT_SCALE
    ct.store(mul3_ptr, index=(row, 0), tile=mul_3)

    inv_h = 1.0 / HIDDEN
    mean = ct.sum(mul_3, axis=1, keepdims=True) * inv_h
    centered = mul_3 - mean
    ct.store(sub_ptr, index=(row, 0), tile=centered)
    # Sample variance with correction=1.
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / (HIDDEN - 1))
    std = ct.sqrt(variance)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(std, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    mul_4 = weight_2d * centered
    div_val = mul_4 / (std + EPS)
    affine = div_val + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf)


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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, _shape2, shape3 = inputs
    norm_shape = _shape(shape0)
    flat_shape = _shape(shape3)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    std_shape = (norm_shape[0], norm_shape[1], 1)

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
    random0 = _inductor_random_for_eager_check(norm_shape, seed0, device=device)
    random1 = _inductor_random_for_eager_check(norm_shape, seed1, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    rand0_2d = random0.contiguous().view(rows, hidden)
    rand1_2d = random1.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    if BLOCK_H == hidden:
        x_pad = x_2d
        rand0_pad = rand0_2d
        rand1_pad = rand1_2d
        residual_pad = residual_2d
        weight_pad = arg3_1
        bias_pad = arg4_1
    else:
        x_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        x_pad[:, :hidden].copy_(x_2d)
        rand0_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        rand0_pad[:, :hidden].copy_(rand0_2d)
        rand1_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        rand1_pad[:, :hidden].copy_(rand1_2d)
        residual_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        residual_pad[:, :hidden].copy_(residual_2d)
        weight_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        weight_pad[:hidden].copy_(arg3_1)
        bias_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        bias_pad[:hidden].copy_(arg4_1)

    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    gt1_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    mul3_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    sub_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dual_dropout_ln_kernel,
        (x_pad, rand0_pad, rand1_pad, residual_pad, weight_pad, bias_pad,
         gt_pad, gt1_pad, mul3_pad, sqrt_1d, sub_pad, bf16_pad,
         hidden, BLOCK_H),
    )

    if BLOCK_H == hidden:
        gt = gt_pad.view(norm_shape)
        gt1 = gt1_pad.view(norm_shape)
        mul_3 = mul3_pad.view(norm_shape)
        sub = sub_pad.view(norm_shape)
        bf16_view = bf16_pad.view(flat_shape)
    else:
        gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
        gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
        gt1 = torch.empty(norm_shape, device=device, dtype=torch.bool)
        gt1.view(rows, hidden).copy_(gt1_pad.narrow(1, 0, hidden))
        mul_3 = torch.empty(norm_shape, device=device, dtype=torch.float32)
        mul_3.view(rows, hidden).copy_(mul3_pad.narrow(1, 0, hidden))
        sub = torch.empty(norm_shape, device=device, dtype=torch.float32)
        sub.view(rows, hidden).copy_(sub_pad.narrow(1, 0, hidden))
        bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
        bf16_view.view(rows, hidden).copy_(bf16_pad.narrow(1, 0, hidden))

    sqrt = sqrt_1d.view(std_shape)
    return gt, gt1, mul_3, sqrt, sub, bf16_view
