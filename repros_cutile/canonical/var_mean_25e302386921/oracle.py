"""cuTile port of var_mean_25e302386921: MegatronBERT dropout LayerNorm.

Pre-generates the seeded random tensor via inductor_random on the Python side,
then a single row-parallel cuTile kernel does dropout, residual add, LN, and
affine. HIDDEN=1024 is a power of two so no padding is required.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 4
BATCH = 16
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 1024
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_layernorm_kernel(
    x_ptr,        # bf16 [ROWS, HIDDEN]
    random_ptr,   # f32 [ROWS, HIDDEN]
    residual_ptr, # f32 [ROWS, HIDDEN]
    weight_ptr,   # f32 [HIDDEN]
    bias_ptr,     # f32 [HIDDEN]
    gt_ptr,       # bool [ROWS, HIDDEN]
    add_ptr,      # f32 [ROWS, HIDDEN]
    norm_ptr,     # f32 [ROWS, HIDDEN]
    out_ptr,      # bf16 [ROWS, HIDDEN]
    div_ptr,      # f32 [ROWS]
    HIDDEN_C: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    # Dropout mask: cast random f32 -> bf16, threshold 0.1 bf16, gt
    random_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    # Dropout apply in bf16, scale in fp32 -> bf16
    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_f = ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE
    scaled_bf = ct.astype(scaled_f, ct.bfloat16)

    x_f = ct.astype(scaled_bf, ct.float32) + residual_f
    ct.store(add_ptr, index=(row, 0), tile=x_f)

    mean = ct.sum(x_f) * (1.0 / HIDDEN_C)
    centered = x_f - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd
    ct.store(norm_ptr, index=(row, 0), tile=normalized)

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_1d = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_1d, (1, BLOCK_H))
    affine_f = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine_f, ct.bfloat16))

    ct.store(div_ptr, index=(row,), tile=invstd * (1.0 / HIDDEN_C))


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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="cfc55f11", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, _shape2 = inputs
    random_shape = _shape_tuple(shape1)
    device = arg0_1.device

    gt = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.bool)
    add = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    normalized = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # Views into the kernel-facing 2D layout
    gt_flat = gt.view(ROWS, HIDDEN)
    add_flat = add.view(ROWS, HIDDEN)
    norm_flat = normalized.view(ROWS, HIDDEN)
    random_flat = random.view(ROWS, HIDDEN)
    residual_flat = arg2_1.reshape(ROWS, HIDDEN)
    div_flat = div.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _dropout_layernorm_kernel,
        (
            arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
            gt_flat, add_flat, norm_flat, out, div_flat,
            HIDDEN, EPS, BLOCK_H,
        ),
    )
    return gt, add, normalized, out, div
