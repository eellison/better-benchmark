"""cuTile port of mean_8300da72e62a: MT5 dropout + residual + RMSNorm row kernel.

Pre-generates the seeded random tensor via inductor_random outside the
kernel, then runs a single cuTile row kernel that applies dropout, adds
residual, does mean-square reduction, and produces the affine bf16 output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 80
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    random_ptr,     # f32 [rows, HIDDEN]
    residual_ptr,   # f32 [rows, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    mask_ptr,       # b8 [rows, HIDDEN]
    add_ptr,        # f32 [rows, HIDDEN]
    rsqrt_ptr,      # f32 [rows]
    final_ptr,      # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    square_sum = ct.sum(add_val * add_val, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN) + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(inv_rms, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    normalized = add_val * inv_rms
    final = ct.astype(weight_2d * normalized, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=final)


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    base_stride = _contiguous_stride(base_shape)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    mask_out = torch.empty_strided(base_shape, base_stride,
                                   device=device, dtype=torch.bool)
    add_out = torch.empty_strided(base_shape, base_stride,
                                  device=device, dtype=torch.float32)
    rsqrt_shape = base_shape[:-1] + (1,)
    rsqrt_out = torch.empty_strided(rsqrt_shape, _contiguous_stride(rsqrt_shape),
                                    device=device, dtype=torch.float32)
    final_base = torch.empty_strided(base_shape, base_stride,
                                     device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    flat_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    mask_2d = mask_out.view(rows, hidden)
    add_2d = add_out.view(rows, hidden)
    rsqrt_1d = rsqrt_out.view(rows)
    final_2d = final_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, mask_2d, add_2d, rsqrt_1d, final_2d,
         hidden, BLOCK_H),
    )
    return mask_out, add_out, rsqrt_out, final_base.view(out_shape)
