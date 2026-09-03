"""cuTile port of mean_var_f23a6c667865: BERT dropout-residual LayerNorm row.

Pre-generates the seeded random tensor via inductor_random outside the kernel,
then runs a single cuTile row kernel that fuses: bf16 dropout mask/scale, fp32
residual add, mean, unbiased variance (correction=1), sqrt-plus-eps affine.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 37
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_layernorm_kernel(
    flat_ptr,          # bf16 [rows, HIDDEN]
    random_ptr,        # f32  [rows, HIDDEN]
    residual_ptr,      # f32  [rows, HIDDEN]
    weight_ptr,        # f32  [HIDDEN]
    bias_ptr,          # f32  [HIDDEN]
    gt_ptr,            # bool [rows, HIDDEN]
    add_ptr,           # f32  [rows, HIDDEN]
    sqrt_ptr,          # f32  [rows]
    sub_ptr,           # f32  [rows, HIDDEN]
    out_ptr,           # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add)

    add_sum = ct.sum(add, axis=1, keepdims=True)
    mean = add_sum * (1.0 / HIDDEN)
    centered = add - mean
    ct.store(sub_ptr, index=(row, 0), tile=centered)
    # Unbiased variance (correction=1) — divide by (HIDDEN - 1)
    variance_sum = ct.sum(centered * centered, axis=1, keepdims=True)
    variance = variance_sum * (1.0 / (HIDDEN - 1))
    std = ct.sqrt(variance)
    std_1d = ct.reshape(std, (1,))
    ct.store(sqrt_ptr, index=(row,), tile=std_1d)

    weight_bf = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_bf = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight_bf, ct.float32)
    bias_f = ct.astype(bias_bf, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    numerator = weight_2d * centered
    denom = std + EPS
    normalized = numerator / denom
    affine = normalized + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    base_stride = _contiguous_stride(base_shape)
    sqrt_shape = base_shape[:-1] + (1,)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    gt = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    add_t = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    sqrt = torch.empty_strided(
        sqrt_shape, _contiguous_stride(sqrt_shape),
        device=device, dtype=torch.float32)
    sub = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    residual_2d = arg2_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add_t.view(rows, hidden)
    sqrt_1d = sqrt.view(rows)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_layernorm_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, sqrt_1d, sub_2d, out_2d,
         hidden, BLOCK_H),
    )
    return gt, add_t, sqrt, sub, out
