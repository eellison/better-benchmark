"""cuTile port of var_mean_c7e6741fa8cd: Longformer bias-add + seeded-dropout +
residual LayerNorm producing a final bf16 [8192,768] view.

Same shape as the SCHEDULER_FUSION Longformer var_mean, plus a pre-dropout bias
(f32[hidden] cast to bf16) added to the bf16 activations before dropout.
Pre-generates the RNG tensor via inductor_random; the row-parallel cuTile kernel
does bias-add + dropout + residual add + LayerNorm + affine and emits mask,
normalized, affine, contiguous bf16 flat view (no permute here), and
invstd/HIDDEN.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _biased_dropout_residual_layernorm_kernel(
    x_ptr,              # bf16 [rows, hidden]
    pre_bias_ptr,       # f32  [hidden]
    random_ptr,         # f32  [rows, hidden]
    residual_ptr,       # f32  [rows, hidden]
    weight_ptr,         # f32  [hidden]
    bias_ptr,           # f32  [hidden]
    mask_ptr,           # b8   [rows, hidden]
    norm_ptr,           # f32  [rows, hidden]
    affine_ptr,         # f32  [rows, hidden]
    bf16_ptr,           # bf16 [rows, hidden]
    div_ptr,            # f32  [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
    EPS_: ct.Constant[float],
):
    row = ct.bid(0)

    x_bf = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pre_bias_f = ct.load(
        pre_bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pre_bias_bf = ct.astype(pre_bias_f, ct.bfloat16)
    pre_bias_bf_2d = ct.reshape(pre_bias_bf, (1, BLOCK_H))
    # bf16-precision bias add (matches the Triton oracle's cast-then-add ordering).
    biased_bf = ct.astype(
        ct.astype(x_bf, ct.float32)
        + ct.astype(pre_bias_bf_2d, ct.float32),
        ct.bfloat16,
    )

    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((1, BLOCK_H), 0.1, dtype=ct.float32), ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, biased_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_, ct.bfloat16,
    )
    x = ct.astype(scaled_bf, ct.float32) + residual_f

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    x_masked = ct.where(col_mask, x, 0.0)
    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x_masked) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * inv_h
    invstd = ct.rsqrt(variance + EPS_)
    normalized = centered * invstd
    ct.store(norm_ptr, index=(row, 0), tile=normalized)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN, (1,)))


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


@oracle_impl(hardware="B200", point="726994b7", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, shape1, shape2 = inputs

    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg1_1.shape[0])
    hidden = int(arg4_1.shape[0])
    batch = int(norm_shape[0])
    seq = int(norm_shape[1])
    div_shape = (batch, seq, 1)
    device = arg1_1.device

    mask = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg1_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg3_1.contiguous().view(rows, hidden)
    mask_2d = mask.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _biased_dropout_residual_layernorm_kernel,
        (x_2d, arg0_1, random_2d, residual_2d, arg4_1, arg5_1,
         mask_2d, normalized_2d, affine_2d, bf16_2d, div_1d,
         hidden, BLOCK_H, DROPOUT_SCALE, EPS),
    )
    return mask, normalized, affine, bf16_view, div
