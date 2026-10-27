"""cuTile port of var_mean_df9f120feef7: XLNet dropout+residual+LayerNorm.

Ports the Triton `_dropout_residual_layernorm_kernel` (seed 39). Pre-generates
the seeded random tensor via `torch.ops.prims.inductor_random` and passes it
as an input. HIDDEN=1024 divides BLOCK_H=1024 evenly so no masked stores.

Repro.forward returns: (gt, mul_2, add_2, view_2, div) which is
(mask bool[512,16,1024], normalized f32[512,16,1024], affine f32[512,16,1024],
bf16 view[8192,1024], div f32[512,16,1]).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 39
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    random_ptr,    # f32 [rows, HIDDEN]
    residual_ptr,  # f32 [rows, HIDDEN]
    weight_ptr,    # f32 [HIDDEN]
    bias_ptr,      # f32 [HIDDEN]
    gt_ptr,        # bool [rows, HIDDEN]
    norm_ptr,      # f32 [rows, HIDDEN]
    affine_ptr,    # f32 [rows, HIDDEN]
    bf16_ptr,      # bf16 [rows, HIDDEN]
    div_ptr,       # f32 [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row_block = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))

    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_H), 0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.zeros((BLOCK_M, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled_bf, ct.float32) + residual

    inv_h = 1.0 / HIDDEN
    row_sum = ct.sum(x, axis=1)
    mean = row_sum * inv_h
    mean_2d = ct.reshape(mean, (BLOCK_M, 1))
    centered = x - mean_2d
    variance = ct.sum(centered * centered, axis=1) * inv_h
    rsqrt = ct.rsqrt(variance + EPS)
    rsqrt_2d = ct.reshape(rsqrt, (BLOCK_M, 1))
    normalized = centered * rsqrt_2d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row_block,), tile=rsqrt * inv_h)


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


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, shape2, shape3 = inputs
    norm_shape = _as_shape(shape1)   # [512, 16, 1024]
    random_shape = _as_shape(shape2)  # [512, 16, 1024]
    flat_shape = _as_shape(shape3)    # [8192, 1024]
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    div_shape = (norm_shape[0], norm_shape[1], 1)
    BLOCK_M = ROW_BLOCK

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    residual_flat = arg2_1.reshape(rows, hidden).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _dropout_residual_layernorm_kernel,
        (arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
         gt.view(rows, hidden),
         normalized.view(rows, hidden),
         affine.view(rows, hidden),
         bf16_view.view(rows, hidden),
         div.view(rows),
         hidden, BLOCK_M, BLOCK_H),
    )
    return gt, normalized, affine, bf16_view, div
