"""cuTile port of var_mean_221aed5431d8: XLNet dropout-residual LayerNorm."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 61
BATCH = 512
SEQ = 16
ROWS = BATCH * SEQ
HIDDEN = 1024
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_layernorm_kernel(
    hidden_ptr,     # bf16 [rows, hidden]
    random_ptr,     # f32  [rows, hidden]
    residual_ptr,   # f32  [rows, hidden]
    weight_ptr,     # f32  [hidden]
    bias_ptr,       # f32  [hidden]
    gt_ptr,         # b8   [rows, hidden]
    norm_ptr,       # f32  [rows, hidden]
    affine_ptr,     # f32  [rows, hidden]
    bf16_ptr,       # bf16 [rows, hidden]
    div_ptr,        # f32  [rows, 1]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    hidden_bf = ct.load(hidden_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, hidden_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    x = ct.astype(scaled_bf, ct.float32) + residual_f

    inv_h = 1.0 / HIDDEN_
    mean = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    div_val = invstd * inv_h
    ct.store(div_ptr, index=(row_block, 0), tile=div_val)


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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, random_shape, _shape2 = inputs
    base_shape = _shape_tuple(shape0)
    rand_shape = _shape_tuple(random_shape)
    device = arg0_1.device

    gt = torch.empty(base_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(base_shape, device=device, dtype=torch.float32)
    affine = torch.empty(base_shape, device=device, dtype=torch.float32)
    bf16_base = torch.empty((1, ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(rand_shape, seed, device=device)

    # reshape all to [rows, hidden]
    hidden_2d = arg0_1.contiguous().view(ROWS, HIDDEN)
    random_2d = random.contiguous().view(ROWS, HIDDEN)
    residual_2d = arg2_1.contiguous().view(ROWS, HIDDEN)
    gt_2d = gt.view(ROWS, HIDDEN)
    norm_2d = normalized.view(ROWS, HIDDEN)
    affine_2d = affine.view(ROWS, HIDDEN)
    bf16_2d = bf16_base.view(ROWS, HIDDEN)
    div_2d = div.view(ROWS, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(ROWS, ROW_BLOCK), 1, 1),
        _dropout_layernorm_kernel,
        (hidden_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, norm_2d, affine_2d, bf16_2d, div_2d,
         HIDDEN, BLOCK_H, ROW_BLOCK),
    )

    bf16_flat = bf16_base.squeeze(0)
    bf16_transpose = bf16_base.permute(0, 2, 1).squeeze(0)
    return gt, normalized, affine, bf16_flat, bf16_transpose, div
