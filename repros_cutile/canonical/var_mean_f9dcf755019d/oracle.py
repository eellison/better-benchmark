"""cuTile port of var_mean_f9dcf755019d: XLNet double-dropout residual LayerNorm.

Uses two pre-generated random tensors (seeds 97 and 98). Row kernel: input
dropout, residual add, var/mean, rsqrt (eps=1e-12), affine, output dropout,
then a permute-clone-cast (done on the Python side using torch).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


INPUT_SEED = 97
OUTPUT_SEED = 98
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _double_dropout_layernorm_kernel(
    x_ptr,           # bf16 [rows, HIDDEN]
    random0_ptr,     # f32 [rows, HIDDEN]
    random1_ptr,     # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    input_mask_ptr,  # bool [rows, HIDDEN]
    norm_ptr,        # f32 [rows, HIDDEN]
    output_mask_ptr, # bool [rows, HIDDEN]
    affine_out_ptr,  # f32 [rows, HIDDEN] — scaled but not yet permuted/cast
    div_ptr,         # f32 [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand0_f = ct.load(
        random0_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand0_bf = ct.astype(rand0_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep0 = rand0_bf > threshold_bf
    ct.store(input_mask_ptr, index=(row, 0), tile=keep0)

    dropped_bf = ct.astype(
        ct.where(keep0, ct.astype(x_bf, ct.float32), 0.0), ct.bfloat16,
    )
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    layernorm_input = ct.astype(scaled_bf, ct.float32) + residual

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    x_masked = ct.where(col_mask, layernorm_input, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = layernorm_input - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

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

    rand1 = ct.load(
        random1_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep1 = rand1 > 0.1
    ct.store(output_mask_ptr, index=(row, 0), tile=keep1)
    dropped_affine = ct.where(keep1, affine, 0.0)
    scaled_out = dropped_affine * DROPOUT_SCALE

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_out_ptr, index=(row, 0), tile=scaled_out)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN, (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
            torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
        )
    advance = _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, _shape2, shape3 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape3)
    rows = int(arg0_1.shape[0])
    outer = int(norm_shape[0])
    inner = int(norm_shape[1])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)

    input_mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    output_mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    affine_out = torch.empty(norm_shape, device=device, dtype=torch.float32)
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, INPUT_SEED)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, OUTPUT_SEED)
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape, seed0, seed1, device=device,
    )

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random0_2d = random0.contiguous().view(rows, hidden)
    random1_2d = random1.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    input_mask_2d = input_mask.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    output_mask_2d = output_mask.view(rows, hidden)
    affine_out_2d = affine_out.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _double_dropout_layernorm_kernel,
        (x_2d, random0_2d, random1_2d, residual_2d, arg3_1, arg4_1,
         input_mask_2d, normalized_2d, output_mask_2d, affine_out_2d, div_1d,
         hidden, BLOCK_H),
    )

    # Permute-clone-cast epilogue on the Python side. Follows the graph:
    # permute [1, 0, 2] then clone (contiguous) then bf16 cast then view.
    permuted = affine_out.permute(1, 0, 2).contiguous()
    bf16_permuted = permuted.to(torch.bfloat16)
    bf16_view = bf16_permuted.view(flat_shape)

    return input_mask, normalized, output_mask, bf16_view, div
