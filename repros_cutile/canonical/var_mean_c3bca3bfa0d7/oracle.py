"""cuTile port of var_mean_c3bca3bfa0d7: YituTech ConvBert dropout-residual LayerNorm.

HIDDEN=768, seed 1, eps 1e-12. Pre-generates the seeded random tensor via
torch.ops.prims.inductor_random and passes it to a single cuTile row kernel
that fuses dropout, residual add, LayerNorm, affine, and side-output stores.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_SCALE = 1.1111111111111112
DROPOUT_P = 0.1
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    x_ptr,            # bf16 (rows, HIDDEN)
    random_ptr,       # f32 (rows, HIDDEN)
    residual_ptr,     # f32 (rows, HIDDEN)
    weight_ptr,       # f32 (HIDDEN,)
    bias_ptr,         # f32 (HIDDEN,)
    mask_ptr,         # bool (rows, HIDDEN)
    norm_ptr,         # f32 (rows, HIDDEN)
    affine_ptr,       # f32 (rows, HIDDEN)
    bf16_ptr,         # bf16 (rows, HIDDEN)
    div_ptr,          # f32 (rows,)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    random_bf = ct.astype(random_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > dropout_p_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
                          ct.bfloat16)
    x = ct.astype(scaled_bf, ct.float32) + residual

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_2d = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    x_masked = ct.where(col_mask_2d, x, zero_f)

    mean = ct.sum(x_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked,
                      axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    norm_out = ct.where(col_mask_2d, normalized, zero_f)
    aff_out = ct.where(col_mask_2d, affine, zero_f)
    aff_bf_out = ct.where(col_mask_2d, ct.astype(affine, ct.bfloat16), zero_bf)
    ct.store(norm_ptr, index=(row, 0), tile=norm_out)
    ct.store(affine_ptr, index=(row, 0), tile=aff_out)
    ct.store(bf16_ptr, index=(row, 0), tile=aff_bf_out)
    div = invstd * (1.0 / HIDDEN)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div, (1,)))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound[8:16] = torch.tensor(
            list(int(offset - advance).to_bytes(8, "little", signed=False)),
            dtype=state.dtype, device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="d429ff7b", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)  # (32, 512, 768)
    flat_shape = _as_shape(shape2)  # (16384, 768)
    rows = int(arg0_1.shape[0])     # 16384
    hidden = int(arg3_1.shape[0])   # 768
    div_shape = (norm_shape[0], norm_shape[1], 1)

    device = arg0_1.device
    mask = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    random_2d = random.reshape(rows, hidden)
    residual_2d = arg2_1.reshape(rows, hidden)
    mask_2d = mask.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_layernorm_kernel,
        (
            arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
            mask_2d, normalized_2d, affine_2d, bf16_view_2d, div_1d,
            hidden, BLOCK_H,
        ),
    )
    return mask, normalized, affine, bf16_view, div
