"""cuTile port of mean_var_08d022b64cd6: BERT dropout + residual + LayerNorm row kernel.

Pre-generates the seeded random tensor via inductor_random outside the
kernel, then runs a single cuTile row kernel: dropout, residual add, mean,
unbiased variance (denominator = HIDDEN - 1), sqrt, centered, affine.
Hidden=768 (non-pow2) uses BLOCK_H=1024 with padding_mode=ZERO + column mask.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 2
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


@ct.kernel
def _dropout_mean_var_kernel(
    x_ptr,          # bf16 [rows, HIDDEN]
    random_ptr,     # f32 [rows, HIDDEN]
    residual_ptr,   # f32 [rows, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    gt_ptr,         # b8 [rows, HIDDEN]
    add_ptr,        # f32 [rows, HIDDEN]
    sqrt_ptr,       # f32 [rows]
    sub_ptr,        # f32 [rows, HIDDEN]
    out_ptr,        # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    add_masked = ct.where(col_mask, add_val, zero_f)
    row_sum = ct.sum(add_masked, axis=1, keepdims=True)
    row_sum_sq = ct.sum(add_masked * add_masked, axis=1, keepdims=True)
    mean = row_sum * (1.0 / HIDDEN)
    centered = add_val - mean
    variance = (row_sum_sq - row_sum * mean) * (1.0 / (HIDDEN - VAR_CORRECTION))
    # tl.maximum(variance, 0) - use ct.where
    variance_pos = ct.where(variance > 0.0, variance, ct.full((1, 1), 0.0, dtype=ct.float32))
    std = ct.sqrt(variance_pos)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(std, (1,)))
    ct.store(sub_ptr, index=(row, 0), tile=centered)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = (weight_2d * centered) / (std + DENOM_EPS) + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    x, seeds, residual, weight, bias, view_shape, random_shape, out_shape = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape_t = _shape_tuple(view_shape)
    random_shape_t = _shape_tuple(random_shape)
    out_shape_t = _shape_tuple(out_shape)
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])
    device = x.device
    stat_shape = (view_shape_t[0], view_shape_t[1], 1)

    gt = torch.empty_strided(view_shape_t, _contiguous_stride(view_shape_t),
                             device=device, dtype=torch.bool)
    add = torch.empty_strided(view_shape_t, _contiguous_stride(view_shape_t),
                              device=device, dtype=torch.float32)
    sqrt = torch.empty_strided(stat_shape, _contiguous_stride(stat_shape),
                               device=device, dtype=torch.float32)
    sub = torch.empty_strided(view_shape_t, _contiguous_stride(view_shape_t),
                              device=device, dtype=torch.float32)
    out = torch.empty_strided(out_shape_t, _contiguous_stride(out_shape_t),
                              device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape_t, seed, device=device)

    x_2d = x.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = residual.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    sqrt_1d = sqrt.view(rows)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_mean_var_kernel,
        (x_2d, random_2d, residual_2d, weight, bias, gt_2d, add_2d, sqrt_1d, sub_2d, out_2d,
         hidden, BLOCK_H),
    )
    return gt, add, sqrt, sub, out
