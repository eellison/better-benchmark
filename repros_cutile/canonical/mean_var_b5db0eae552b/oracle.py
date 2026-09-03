"""cuTile port of mean_var_b5db0eae552b: BERT dual seeded dropout + LayerNorm.

Uses pre-generated random tensors (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. HIDDEN=768 requires BLOCK_H=1024
with masked reductions.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 9
SEED_INDEX_1 = 10
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 768
BLOCK_H = 1024
EPS = 1.0e-6


@ct.kernel
def _dual_dropout_layernorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random0_ptr,     # f32  [rows, HIDDEN]
    random1_ptr,     # f32  [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    bias_ptr,        # f32  [HIDDEN]
    gt0_ptr,         # b8   [rows, BLOCK_H]  (padded)
    gt1_ptr,         # b8   [rows, BLOCK_H]
    dropped_ptr,     # f32  [rows, BLOCK_H]
    sqrt_ptr,        # f32  [rows]
    sub_ptr,         # f32  [rows, BLOCK_H]
    out_ptr,         # bf16 [rows, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    flat_bf = ct.load(
        flat_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random0 = ct.load(
        random0_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random1 = ct.load(
        random1_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    threshold_bf = ct.astype(
        ct.full((ROW_BLOCK, BLOCK_H_), 0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep0 = ct.astype(random0, ct.bfloat16) > threshold_bf
    keep1 = random1 > 0.1
    ct.store(gt0_ptr, index=(row_block, 0), tile=keep0)
    ct.store(gt1_ptr, index=(row_block, 0), tile=keep1)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.bfloat16)
    dropped0_bf = ct.where(keep0, flat_bf, zero_bf)
    scaled0_bf = ct.astype(
        ct.astype(dropped0_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    add = residual + ct.astype(scaled0_bf, ct.float32)
    dropped1 = ct.where(keep1, add, ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.float32))
    x = dropped1 * DROPOUT_SCALE
    ct.store(dropped_ptr, index=(row_block, 0), tile=x)

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.float32))
    mean = ct.sum(x_masked, axis=1) * (1.0 / HIDDEN_)
    mean_2d = ct.reshape(mean, (ROW_BLOCK, 1))
    centered = x - mean_2d
    centered_masked = ct.where(col_mask, centered, ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.float32))
    var_sum = ct.sum(centered_masked * centered_masked, axis=1)
    variance = var_sum * (1.0 / (HIDDEN_ - 1))
    std = ct.sqrt(variance)
    ct.store(sqrt_ptr, index=(row_block,), tile=std)
    ct.store(sub_ptr, index=(row_block, 0), tile=centered)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    numerator = weight_2d * centered
    std_2d = ct.reshape(std, (ROW_BLOCK, 1))
    normalized = numerator / (std_2d + EPS)
    affine = normalized + bias_2d
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))


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


def _random_advance(shape, *, device):
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


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    advance = _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


@oracle_impl(hardware="B200", point="4205ff34", ROW_BLOCK=1)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2, shape3 = inputs
    del _shape1
    base_shape = _shape_tuple(shape0)  # [16,128,768]
    random_shape = _shape_tuple(shape2)  # [16,128,768]
    out_shape = _shape_tuple(shape3)  # [2048,768]
    base_stride = _contiguous_stride(base_shape)
    sqrt_shape = base_shape[:-1] + (1,)

    gt0 = torch.empty_strided(
        base_shape, base_stride,
        device=arg0_1.device, dtype=torch.bool,
    )
    gt1 = torch.empty_strided(
        base_shape, base_stride,
        device=arg0_1.device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        base_shape, base_stride,
        device=arg0_1.device, dtype=torch.float32,
    )
    sqrt = torch.empty_strided(
        sqrt_shape, _contiguous_stride(sqrt_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    sub = torch.empty_strided(
        base_shape, base_stride,
        device=arg0_1.device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape, seed0, seed1, device=arg0_1.device
    )

    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN

    # Padded per-kernel outputs (BLOCK_H = 1024 wide), we then narrow.
    gt0_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bool)
    gt1_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bool)
    dropped_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    sub_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    out_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)

    flat_2d = arg0_1  # already [2048,768]
    residual_2d = arg2_1.reshape(rows, hidden)
    random0_2d = random0.reshape(rows, hidden).contiguous()
    random1_2d = random1.reshape(rows, hidden).contiguous()
    sqrt_1d = sqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dual_dropout_layernorm_kernel,
        (flat_2d, random0_2d, random1_2d, residual_2d, arg3_1, arg4_1,
         gt0_pad, gt1_pad, dropped_pad, sqrt_1d, sub_pad, out_pad,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    # Copy valid columns back into the final outputs.
    gt0.view(rows, hidden).copy_(gt0_pad.narrow(1, 0, hidden))
    gt1.view(rows, hidden).copy_(gt1_pad.narrow(1, 0, hidden))
    dropped.view(rows, hidden).copy_(dropped_pad.narrow(1, 0, hidden))
    sub.view(rows, hidden).copy_(sub_pad.narrow(1, 0, hidden))
    out.view(rows, hidden).copy_(out_pad.narrow(1, 0, hidden))

    return gt0, gt1, dropped, sqrt, sub, out
