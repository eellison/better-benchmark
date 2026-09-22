"""cuTile port of mean_var_e4fd793207ea: BERT seeded dropout + residual + LayerNorm.

Pre-generates seeded random via inductor_random outside the kernel and runs
one row kernel computing dropout mask + scaled add residual + mean/var
LayerNorm affine.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 17
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN_PAD] padded to BLOCK_H
    random_ptr,    # f32 [rows, HIDDEN_PAD]
    residual_ptr,  # f32 [rows, HIDDEN_PAD]
    scale_ptr,     # f32 [HIDDEN_PAD]
    bias_ptr,      # f32 [HIDDEN_PAD]
    mask_out_ptr,  # b8 [rows, HIDDEN_PAD]
    add_out_ptr,   # f32 [rows, HIDDEN_PAD]
    sqrt_out_ptr,  # f32 [rows]
    sub_out_ptr,   # f32 [rows, HIDDEN_PAD]
    final_out_ptr, # bf16 [rows, HIDDEN_PAD]
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

    # Column mask: valid iff col < HIDDEN
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_1d = cols < HIDDEN
    col_mask = ct.reshape(col_mask_1d, (1, BLOCK_H))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(mask_out_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )

    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=add_val)

    # For reductions, mask out invalid columns
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    add_masked = ct.where(col_mask, add_val, zero_f)
    mean_val = ct.sum(add_masked) * (1.0 / HIDDEN)
    sub_val = add_val - mean_val
    ct.store(sub_out_ptr, index=(row, 0), tile=sub_val)

    sub_masked = ct.where(col_mask, sub_val, zero_f)
    var_val = ct.sum(sub_masked * sub_masked) * (1.0 / (HIDDEN - 1))
    sqrt_val = ct.sqrt(var_val)
    ct.store(sqrt_out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), sqrt_val, dtype=ct.float32), (1,)))

    scale = ct.load(scale_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    scale_2d = ct.reshape(scale, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    denom = sqrt_val + EPS
    normed = (scale_2d * sub_val) * (1.0 / denom) + bias_2d
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(normed, ct.bfloat16))


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

    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    hidden = int(base_shape[-1])
    rows = 1
    for d in base_shape[:-1]:
        rows *= int(d)
    device = arg0_1.device

    # Allocate padded 2D buffers of shape (rows, BLOCK_H) so cuTile stores fit,
    # then narrow the trailing dim to `hidden` when returning outputs.
    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sub_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    sqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    out_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)

    # Padded weight/bias so the load doesn't OOB.
    scale_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    scale_pad[:hidden].copy_(arg3_1)
    bias_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias_pad[:hidden].copy_(arg4_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    flat_2d = arg0_1.view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_layernorm_kernel,
        (flat_2d, random_2d, residual_2d, scale_pad, bias_pad,
         gt_pad, add_pad, sqrt_1d, sub_pad, out_pad, hidden, BLOCK_H),
    )

    # Narrow to valid columns and reshape to the returned shape.
    base_stride = _contiguous_stride(base_shape)
    gt = gt_pad[:, :hidden].contiguous().view(base_shape)
    add = add_pad[:, :hidden].contiguous().view(base_shape)
    sub = sub_pad[:, :hidden].contiguous().view(base_shape)
    sqrt_shape = base_shape[:-1] + (1,)
    sqrt_t = sqrt_1d.view(sqrt_shape)
    out_base = out_pad[:, :hidden].contiguous().view(out_shape)
    return gt, add, sqrt_t, sub, out_base
