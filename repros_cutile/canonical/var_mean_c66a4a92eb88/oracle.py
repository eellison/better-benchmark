"""cuTile port of var_mean_c66a4a92eb88: GPT-2 dropout+LayerNorm.

Outputs: (gt, add=residual+dropout, mul_2=normalized, bf16_flat, permute, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 19
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_layernorm_kernel(
    hidden_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    norm_ptr,
    bf16_ptr,
    div_ptr,
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    EPS_: ct.Constant[float],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)

    hidden_v = ct.load(hidden_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)

    cols_i32 = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_1d = cols_i32 < HIDDEN_
    col_mask = ct.reshape(col_mask_1d, (1, BLOCK_H))

    rand_bf = ct.astype(rand, ct.bfloat16)
    thresh_bf = ct.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, hidden_v, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_, ct.bfloat16)
    residual_f = ct.astype(residual, ct.float32)
    x_raw = ct.astype(scaled_bf, ct.float32) + residual_f
    zero_f = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.float32)
    x = ct.where(col_mask, x_raw, zero_f)
    ct.store(add_ptr, index=(row_block, 0), tile=x)

    inv_h = 1.0 / HIDDEN_
    mean_1d = ct.sum(x, axis=1, keepdims=True) * inv_h
    centered = ct.where(col_mask, x - mean_1d, zero_f)
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS_)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(gt_ptr, index=(row_block, 0), tile=keep)
    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row_block, 0), tile=invstd_1d * inv_h)


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
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
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


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="bf8decda", ROW_BLOCK=1)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    hidden_input, seeds, residual, weight, bias, shape0, random_shape, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    rand_shape = _shape_tuple(random_shape)
    batch = int(base_shape[0])
    seq = int(base_shape[1])
    hidden = int(base_shape[2])
    rows = batch * seq
    device = hidden_input.device
    BLOCK_H = _next_pow2(hidden)

    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div = torch.empty_strided((batch, seq, 1), (seq, 1, 1),
                              device=device, dtype=torch.float32)

    hidden_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    hidden_pad[:, :hidden].copy_(hidden_input.view(rows, hidden))
    residual_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    residual_pad[:, :hidden].copy_(residual.contiguous().view(rows, hidden))
    weight_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_pad[:hidden].copy_(weight)
    bias_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias_pad[:hidden].copy_(bias)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(rand_shape, seed, device=device)
    random_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    random_pad[:, :hidden].copy_(random.contiguous().view(rows, hidden))

    div_2d = div.view(rows, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_layernorm_kernel,
        (hidden_pad, random_pad, residual_pad, weight_pad, bias_pad,
         gt_pad, add_pad, norm_pad, bf16_pad, div_2d,
         hidden, BLOCK_H, ROW_BLOCK, EPS, DROPOUT_P, DROPOUT_SCALE),
    )

    gt = gt_pad[:, :hidden].reshape(base_shape)
    add = add_pad[:, :hidden].reshape(base_shape)
    normalized = norm_pad[:, :hidden].reshape(base_shape)
    bf16_base = bf16_pad[:, :hidden].contiguous().view(rows, hidden)
    permute = bf16_base.permute(1, 0)
    return gt, add, normalized, bf16_base, permute, div
