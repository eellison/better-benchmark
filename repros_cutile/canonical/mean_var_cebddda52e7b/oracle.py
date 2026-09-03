"""cuTile port of mean_var_cebddda52e7b: BERT seeded-dropout residual LayerNorm.

Pre-generates seeded random via inductor_random, then a single cuTile row
kernel: bf16 dropout mask, dropout scale, f32 residual add, LN (mean +
var(correction=1) + rsqrt), affine, final bf16.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 32
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr,       # bf16 (rows, HIDDEN)
    random_ptr,     # f32 (rows, HIDDEN)
    residual_ptr,   # f32 (rows, HIDDEN)
    weight_ptr,     # f32 (HIDDEN,)
    bias_ptr,       # f32 (HIDDEN,)
    gt_ptr,         # bool (rows, HIDDEN)
    add_ptr,        # f32 (rows, HIDDEN)
    sqrt_ptr,       # f32 (rows,)
    sub_ptr,        # f32 (rows, HIDDEN)
    out_ptr,        # bf16 (rows, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                        padding_mode=ct.PaddingMode.ZERO)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                      padding_mode=ct.PaddingMode.ZERO)
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid_1d = cols < HIDDEN
    col_valid = ct.reshape(col_valid_1d, (1, BLOCK_H))

    # Store gt (bool) via scatter — only write valid columns.
    cols_2d = ct.reshape(row * HIDDEN + cols, (1, BLOCK_H))
    ct.scatter(gt_ptr, cols_2d, keep, mask=col_valid)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
                          ct.bfloat16)
    added = residual + ct.astype(scaled_bf, ct.float32)
    ct.scatter(add_ptr, cols_2d, added, mask=col_valid)

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    added_masked = ct.where(col_valid, added, zero_f)
    mean = ct.sum(added_masked, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = added - mean
    centered_masked = ct.where(col_valid, centered, zero_f)
    ct.scatter(sub_ptr, cols_2d, centered, mask=col_valid)

    # Unbiased var with correction=1.
    var = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / (HIDDEN - 1))
    sqrt_val = ct.sqrt(var)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(sqrt_val, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                      padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                    padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    mul_2 = weight_2d * centered
    denom = sqrt_val + EPS
    div = mul_2 / denom
    add_2 = div + bias_2d
    out_bf = ct.astype(add_2, ct.bfloat16)
    ct.scatter(out_ptr, cols_2d, out_bf, mask=col_valid)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape(shape):
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape(shape0)
    random_shape = _shape(shape1)
    flat_shape = _shape(shape2)
    hidden = int(arg3_1.shape[0])
    n_rows = int(arg0_1.shape[0])
    stat_shape = full_shape[:-1] + (1,)

    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                              device=arg0_1.device, dtype=torch.bool)
    add = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                               device=arg0_1.device, dtype=torch.float32)
    sqrt_out = torch.empty_strided(stat_shape, _contiguous_stride(stat_shape),
                                    device=arg0_1.device, dtype=torch.float32)
    sub_out = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                                    device=arg0_1.device, dtype=torch.float32)
    out_base = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                                    device=arg0_1.device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed,
                                              device=arg0_1.device)

    flat_2d = arg0_1.contiguous().view(n_rows, hidden)
    residual_2d = arg2_1.view(n_rows, hidden)
    random_2d = random.contiguous().view(n_rows, hidden)
    total_elems = n_rows * hidden
    gt_flat = gt.view(total_elems)
    add_flat = add.view(total_elems)
    sub_flat = sub_out.view(total_elems)
    sqrt_1d = sqrt_out.view(n_rows)
    out_flat = out_base.view(total_elems)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _dropout_residual_ln_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_flat, add_flat, sqrt_1d, sub_flat, out_flat,
         hidden, BLOCK_H),
    )

    return gt, add, sqrt_out, sub_out, out_base.view(flat_shape)
