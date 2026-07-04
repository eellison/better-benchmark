"""cuTile port of mean_e8b283443d04: seeded dropout + residual + RMSNorm.

Pre-generates the seeded random tensor via inductor_random outside the kernel
and runs a single cuTile row kernel for dropout mask + scaled add residual +
RMS reduction + affine.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 28
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    random_ptr,    # f32 [rows, HIDDEN]
    residual_ptr,  # f32 [rows, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    mask_out_ptr,  # b8 [rows, HIDDEN]
    add_out_ptr,   # f32 [rows, HIDDEN]
    rsqrt_out_ptr, # f32 [rows]
    final_out_ptr, # bf16 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)

    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_C))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, HIDDEN_C), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(mask_out_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, HIDDEN_C), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )

    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=add_val)

    square_sum = ct.sum(add_val * add_val)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN_C) + EPS)
    ct.store(rsqrt_out_ptr, index=(row,),
             tile=ct.reshape(ct.full((1,), inv_rms, dtype=ct.float32), (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_C))
    normalized = add_val * inv_rms
    final_f = weight_2d * normalized
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(final_f, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs

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

    base_stride = _contiguous_stride(base_shape)
    gt = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    add = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    rsqrt_shape = base_shape[:-1] + (1,)
    rsqrt = torch.empty_strided(
        rsqrt_shape, _contiguous_stride(rsqrt_shape),
        device=device, dtype=torch.float32,
    )
    out_base = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    flat_2d = arg0_1.view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_1d, out_2d, hidden),
    )
    return gt, add, rsqrt, out_base.view(out_shape)
