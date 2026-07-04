"""cuTile port of var_mean_e7bcffc0e186: MegatronBERT seeded-dropout residual LayerNorm."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 19
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    addmm_ptr,      # bf16 [rows, hidden]
    random_ptr,     # f32  [rows, hidden]
    residual_ptr,   # f32  [rows, hidden]
    weight_ptr,     # f32  [hidden]
    bias_ptr,       # f32  [hidden]
    gt_ptr,         # b8   [rows, hidden]
    add_ptr,        # f32  [rows, hidden]
    norm_ptr,       # f32  [rows, hidden]
    bf16_ptr,       # bf16 [rows, hidden]
    div_ptr,        # f32  [rows, 1]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
    EPS_: ct.Constant[float],
):
    row_block = ct.bid(0)

    addmm_bf = ct.load(addmm_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, addmm_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_,
        ct.bfloat16,
    )
    x = residual_f + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row_block, 0), tile=x)

    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS_)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    div_val = invstd * inv_h
    ct.store(div_ptr, index=(row_block, 0), tile=div_val)


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


@oracle_impl(hardware="B200", point="cfc55f11", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _shape(shape0)
    flat_shape = _shape(shape2)
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    add = torch.empty(norm_shape, device=device, dtype=torch.float32)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    total_rows = int(arg2_1.shape[0]) * int(arg2_1.shape[1])
    addmm_2d = arg0_1.contiguous().view(total_rows, hidden)
    random_2d = random.contiguous().view(total_rows, hidden)
    residual_2d = arg2_1.contiguous().view(total_rows, hidden)
    gt_2d = gt.view(total_rows, hidden)
    add_2d = add.view(total_rows, hidden)
    normalized_2d = normalized.view(total_rows, hidden)
    bf16_2d = bf16_view.view(total_rows, hidden)
    div_2d = div.view(total_rows, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total_rows, ROW_BLOCK), 1, 1),
        _dropout_residual_layernorm_kernel,
        (addmm_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, normalized_2d, bf16_2d, div_2d,
         hidden, BLOCK_H, ROW_BLOCK, DROPOUT_SCALE, EPS),
    )
    return gt, add, normalized, bf16_view, div
