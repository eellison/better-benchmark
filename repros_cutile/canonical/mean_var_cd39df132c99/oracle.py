"""cuTile port of mean_var_cd39df132c99: BERT dropout + residual + dropout + LayerNorm.

Pre-generates two seeded randoms via inductor_random (seed 44 for bf16-thresh
dropout on x; seed 45 for fp32-thresh dropout on residual-add), then a single
row cuTile kernel handles: bf16 first dropout, fp32 residual add, fp32 second
dropout, fused row mean + unbiased variance (correction=1), affine LayerNorm
with sqrt(var)+1e-6 denominator, bf16 output.

hidden=768 is not a power of two so the kernel uses ct.gather with a col mask
+ padding to a next power-of-two BLOCK_H (=1024). All reductions/stores are
masked on the valid columns.

Returns (gt0, gt1, dropped, sqrt, sub, view).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 44
SEED_INDEX_1 = 45
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


@ct.kernel
def _dropout_dropout_layernorm_kernel(
    x_ptr,        # bf16 [rows, HIDDEN]
    random0_ptr,  # f32  [rows, HIDDEN]
    random1_ptr,  # f32  [rows, HIDDEN]
    residual_ptr, # f32  [rows, HIDDEN]
    weight_ptr,   # f32  [HIDDEN]
    bias_ptr,     # f32  [HIDDEN]
    gt0_ptr,      # b8   [rows, HIDDEN]
    gt1_ptr,      # b8   [rows, HIDDEN]
    dropped_ptr,  # f32  [rows, HIDDEN]
    sqrt_ptr,     # f32  [rows]
    sub_ptr,      # f32  [rows, HIDDEN]
    out_ptr,      # bf16 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN_C
    row_bcast = ct.full((BLOCK_H,), row, dtype=ct.int32)

    x = ct.gather(x_ptr, (row_bcast, cols), mask=col_valid, padding_value=0)
    rand0_f = ct.gather(random0_ptr, (row_bcast, cols), mask=col_valid, padding_value=0)
    rand1 = ct.gather(random1_ptr, (row_bcast, cols), mask=col_valid, padding_value=0)
    residual = ct.gather(residual_ptr, (row_bcast, cols), mask=col_valid, padding_value=0)

    # First dropout (bf16 threshold).
    rand0_bf = ct.astype(rand0_f, ct.bfloat16)
    p_bf = ct.full((BLOCK_H,), DROPOUT_P, dtype=ct.bfloat16)
    keep0 = rand0_bf > p_bf
    ct.scatter(gt0_ptr, (row_bcast, cols), keep0, mask=col_valid)

    zero_bf = ct.full((BLOCK_H,), 0.0, dtype=ct.bfloat16)
    first_dropped = ct.where(keep0, x, zero_bf)
    first_scaled = ct.astype(ct.astype(first_dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(first_scaled, ct.float32)

    # Second dropout (fp32 threshold).
    p_f = ct.full((BLOCK_H,), DROPOUT_P, dtype=ct.float32)
    keep1 = rand1 > p_f
    ct.scatter(gt1_ptr, (row_bcast, cols), keep1, mask=col_valid)

    zero_f = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    norm_input = ct.where(keep1, add, zero_f) * DROPOUT_SCALE
    ct.scatter(dropped_ptr, (row_bcast, cols), norm_input, mask=col_valid)

    # Row mean + unbiased variance (Bessel's correction=1), masked.
    masked_input = ct.where(col_valid, norm_input, zero_f)
    row_sum = ct.sum(masked_input)
    mean = row_sum * (1.0 / HIDDEN_C)
    centered = norm_input - mean
    ct.scatter(sub_ptr, (row_bcast, cols), centered, mask=col_valid)

    masked_centered = ct.where(col_valid, centered, zero_f)
    sq_sum = ct.sum(masked_centered * masked_centered)
    variance = sq_sum * (1.0 / (HIDDEN_C - VAR_CORRECTION))
    std = ct.sqrt(variance)
    ct.store(sqrt_ptr, index=(row,),
             tile=ct.full((1,), std, dtype=ct.float32))

    weight = ct.gather(weight_ptr, cols, mask=col_valid, padding_value=0)
    bias = ct.gather(bias_ptr, cols, mask=col_valid, padding_value=0)
    denom = std + DENOM_EPS
    affine = (weight * centered) * (1.0 / denom) + bias
    ct.scatter(out_ptr, (row_bcast, cols), ct.astype(affine, ct.bfloat16), mask=col_valid)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_advance(shape, *, device):
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
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _inductor_random_for_eager_check(shape, seed, *, device, calls_back):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
    advance = _inductor_random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewind = advance * calls_back
    if offset >= rewind:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - rewind)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    x, seeds, residual, weight, bias, view_shape, random0_shape, random1_shape, out_shape = inputs

    view_shape = _shape_tuple(view_shape)
    random0_shape = _shape_tuple(random0_shape)
    random1_shape = _shape_tuple(random1_shape)
    out_shape = _shape_tuple(out_shape)
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])
    stat_shape = (view_shape[0], view_shape[1], 1)
    device = x.device

    gt0 = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    gt1 = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    sqrt_out = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=device, dtype=torch.float32,
    )
    sub = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
    random0 = _inductor_random_for_eager_check(
        random0_shape, seed0, device=device, calls_back=2,
    )
    random1 = _inductor_random_for_eager_check(
        random1_shape, seed1, device=device, calls_back=1,
    )

    x_2d = x.view(rows, hidden)
    random0_2d = random0.view(rows, hidden)
    random1_2d = random1.view(rows, hidden)
    residual_2d = residual.view(rows, hidden)
    gt0_2d = gt0.view(rows, hidden)
    gt1_2d = gt1.view(rows, hidden)
    dropped_2d = dropped.view(rows, hidden)
    sqrt_1d = sqrt_out.view(rows)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_dropout_layernorm_kernel,
        (x_2d, random0_2d, random1_2d, residual_2d, weight, bias,
         gt0_2d, gt1_2d, dropped_2d, sqrt_1d, sub_2d, out_2d,
         hidden, BLOCK_H),
    )

    return gt0, gt1, dropped, sqrt_out, sub, out
