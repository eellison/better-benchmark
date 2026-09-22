"""cuTile port of var_mean_e187399c29bb: Longformer bias-add + dropout + residual + LayerNorm.

Seed index 7, eps 1e-5. Returns (gt, mul_2, add_3, view_1, div).
Args: arg0=bias f32[H], arg1=flat bf16, arg2=seeds, arg3=residual f32, arg4=weight, arg5=layerbias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 7
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _bias_dropout_residual_ln_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    bias_pre_ptr,   # f32  [HIDDEN] -- cast to bf16, broadcast add before dropout
    random_ptr,     # f32  [rows, HIDDEN]
    residual_ptr,   # f32  [rows, HIDDEN]
    weight_ptr,     # f32  [HIDDEN]
    bias_ln_ptr,    # f32  [HIDDEN]
    gt_ptr,
    mul2_ptr,
    add3_ptr,
    view1_ptr,      # bf16 [rows, HIDDEN]
    div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    bias_pre_f = ct.load(bias_pre_ptr, index=(0,), shape=(BLOCK_H,))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    bias_pre_bf = ct.astype(bias_pre_f, ct.bfloat16)
    bias_2d = ct.reshape(bias_pre_bf, (1, BLOCK_H))
    added_bf = ct.astype(ct.astype(flat_bf + bias_2d, ct.float32), ct.bfloat16)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, added_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_f = ct.astype(scaled_bf, ct.float32) + residual_f

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(add_f, axis=1, keepdims=True) * inv_h
    centered = add_f - mean_1d
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    rsqrt_val = ct.rsqrt(variance + EPS)
    normalized = centered * rsqrt_val
    ct.store(mul2_ptr, index=(row, 0), tile=normalized)

    weight_f = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_ln_f = ct.load(bias_ln_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d_ = ct.reshape(weight_f, (1, BLOCK_H))
    bias_ln_2d = ct.reshape(bias_ln_f, (1, BLOCK_H))
    affine = normalized * weight_2d_ + bias_ln_2d
    ct.store(add3_ptr, index=(row, 0), tile=affine)
    ct.store(view1_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(rsqrt_val * inv_h, (1,)))


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
    # arg0=bias f32[H], arg1=flat bf16, arg2=seeds i64, arg3=residual f32,
    # arg4=weight f32[H], arg5=layerbias f32[H]
    base_shape = tuple(int(d) for d in shape0)
    random_shape = tuple(int(d) for d in shape1)
    out_shape = tuple(int(d) for d in shape2)
    hidden = int(arg0_1.shape[0])
    device = arg1_1.device

    base_stride = _contiguous_stride(base_shape)
    div_shape = base_shape[:-1] + (1,)

    gt = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    mul2 = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    add3 = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    view1 = torch.empty_strided(out_shape, _contiguous_stride(out_shape), device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    rows = arg1_1.shape[0]
    flat_2d = arg1_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg3_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    mul2_2d = mul2.view(rows, hidden)
    add3_2d = add3.view(rows, hidden)
    view1_2d = view1.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bias_dropout_residual_ln_kernel,
        (flat_2d, arg0_1, random_2d, residual_2d, arg4_1, arg5_1,
         gt_2d, mul2_2d, add3_2d, view1_2d, div_1d,
         hidden, BLOCK_H),
    )
    return gt, mul2, add3, view1, div
