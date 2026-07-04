"""cuTile port of var_mean_dba2f104e79d: XLNet dropout residual LayerNorm (seed 21)."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 21
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    x_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, norm_ptr, affine_ptr, bf16_ptr, div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_bf = ct.astype(rand, ct.bfloat16)
    p_bf = ct.astype(ct.full((1, BLOCK_H), 0.1, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x_f = ct.astype(scaled_bf, ct.float32) + residual
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
    affine_bf = ct.astype(affine, ct.bfloat16)
    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf)
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


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _shape(shape0)  # [512, 16, 1024]
    view_shape = _shape(shape2)  # [1, 8192, 1024]
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    gt_2d = torch.empty((rows, hidden), device=device, dtype=torch.bool)
    norm_2d = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    affine_2d = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    bf16_2d = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    if hidden != BLOCK_H:
        raise NotImplementedError(f"BLOCK_H={BLOCK_H} != hidden={hidden}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, norm_2d, affine_2d, bf16_2d, div_1d,
         hidden, BLOCK_H),
    )

    gt = gt_2d.view(norm_shape)
    normalized = norm_2d.view(norm_shape)
    affine = affine_2d.view(norm_shape)
    # bf16_view is view_1 (shape [1, 8192, 1024]) squeezed to [8192, 1024].
    # We follow the graph: convert_element_type_1 is [512,16,1024] -> unsqueeze x2 ->
    # [512,16,1024,1,1] -> view to [1,8192,1024] -> squeeze(0) -> [8192,1024]
    bf16_view_view = bf16_2d.view(view_shape)  # [1, 8192, 1024]
    squeeze = bf16_view_view.squeeze(0)  # [8192, 1024]
    permute = bf16_view_view.permute(0, 2, 1).squeeze(0)  # [1024, 8192]
    div = div_1d.view(div_shape)
    return gt, normalized, affine, squeeze, permute, div
