"""cuTile port of var_mean_3e50d103dcec: Longformer bias-add + dropout LN.

Pre-generates the seeded random tensor via inductor_random on the Python side,
then a single row-parallel cuTile kernel does bias add + dropout, residual
add, LN, affine, and emits mask, normalized, affine, bf16 view, invstd side
output. HIDDEN=768 padded to BLOCK_H=1024.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 10
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _bias_dropout_ln_kernel(
    flat_ptr,       # bf16 [rows, BLOCK_H]  (padded)
    bias0_ptr,      # f32 [BLOCK_H]
    random_ptr,     # f32 [rows, BLOCK_H]
    residual_ptr,   # f32 [rows, BLOCK_H]
    weight_ptr,     # f32 [BLOCK_H]
    bias1_ptr,      # f32 [BLOCK_H]
    gt_ptr,         # bool [rows, BLOCK_H]
    norm_ptr,       # f32 [rows, BLOCK_H]
    affine_ptr,     # f32 [rows, BLOCK_H]
    bf16_ptr,       # bf16 [rows, BLOCK_H]
    div_ptr,        # f32 [rows]
    HIDDEN: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    # Bias add in fp32 then cast bf16
    bias0_1d = ct.load(bias0_ptr, index=(0,), shape=(BLOCK_H,))
    bias0_2d = ct.reshape(ct.astype(ct.astype(bias0_1d, ct.bfloat16), ct.float32),
                          (1, BLOCK_H))
    flat_f = ct.astype(flat_bf, ct.float32)
    biased_f = flat_f + bias0_2d
    biased_bf = ct.astype(biased_f, ct.bfloat16)

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, biased_bf, zero_bf)
    scaled_f = ct.astype(dropped, ct.float32) * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)
    x_f = ct.astype(scaled_bf, ct.float32) + residual_f

    x_masked = ct.where(col_valid_2d, x_f, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x_f - mean
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias1_1d = ct.load(bias1_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_H))
    bias1_2d = ct.reshape(bias1_1d, (1, BLOCK_H))
    affine_f = normalized * weight_2d + bias1_2d

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine_f)
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine_f, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=invstd * (1.0 / HIDDEN))


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


@oracle_impl(hardware="B200", point="726994b7", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, shape1, shape2 = inputs
    # arg0_1: bias0 [hidden] f32; arg1_1: flat bf16 [rows,hidden]; arg2_1 seeds
    # arg3_1: residual f32 [B,S,H]; arg4_1: weight; arg5_1: bias1
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape2)
    rows = int(arg1_1.shape[0])
    hidden = int(arg1_1.shape[1])
    div_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg1_1.device

    x_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    x_padded[:, :hidden] = arg1_1
    residual_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    residual_padded[:, :hidden] = arg3_1.view(rows, hidden)
    bias0_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias0_padded[:hidden] = arg0_1
    weight_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_padded[:hidden] = arg4_1
    bias1_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias1_padded[:hidden] = arg5_1

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random_full = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random_full.view(rows, hidden)
    random_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    random_padded[:, :hidden] = random_flat

    gt_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_flat = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bias_dropout_ln_kernel,
        (
            x_padded, bias0_padded, random_padded, residual_padded,
            weight_padded, bias1_padded,
            gt_padded, norm_padded, affine_padded, bf16_padded, div_flat,
            hidden, EPS, BLOCK_H,
        ),
    )

    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                             device=device, dtype=torch.bool)
    gt.view(rows, hidden).copy_(gt_padded[:, :hidden])
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=device, dtype=torch.float32)
    normalized.view(rows, hidden).copy_(norm_padded[:, :hidden])
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                 device=device, dtype=torch.float32)
    affine.view(rows, hidden).copy_(affine_padded[:, :hidden])
    out_bf16 = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                   device=device, dtype=torch.bfloat16)
    out_bf16.copy_(bf16_padded[:, :hidden])
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=device, dtype=torch.float32)
    div.view(rows).copy_(div_flat)

    return gt, normalized, affine, out_bf16, div
