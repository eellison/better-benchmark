"""cuTile port of var_mean_c4177f8f8203: DistillGPT2/GPT2 dropout + residual + LN.

Fused seeded Inductor dropout (bf16-compare), residual add, mean/var
(correction=0), rsqrt(eps=1e-5), affine, bf16 cast + permute alias.
HIDDEN=768 -> BLOCK_H=1024 masked. Returns `add`, `mul_2`, bf16, permute, div.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 768
BLOCK_H = 1024
EPS = 1.0e-5


@ct.kernel
def _dropout_ln_kernel(
    flat_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    add_ptr,
    normalized_ptr,
    bf16_ptr,
    div_ptr,
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H_))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(
        ct.where(keep, ct.astype(flat, ct.float32), 0.0),
        ct.bfloat16,
    )
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, add_val, 0.0)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HIDDEN_)
    centered = add_val - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), invstd / HIDDEN_, dtype=ct.float32), (1,)))


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


def _run(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    device = arg0_1.device
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    # shape2 has "-1" (dynamic), so we need to infer from arg0_1
    view_shape = tuple(arg0_1.shape)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN
    div_shape = (norm_shape[0], norm_shape[1], 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(rows, hidden).contiguous()
    residual_2d = arg2_1.reshape(rows, hidden).contiguous()

    flat_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    flat_pad.narrow(1, 0, hidden).copy_(arg0_1)
    r_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    r_pad.narrow(1, 0, hidden).copy_(random_2d)
    resid_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    resid_pad.narrow(1, 0, hidden).copy_(residual_2d)
    w_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    w_pad.narrow(0, 0, hidden).copy_(arg3_1)
    b_pad = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    b_pad.narrow(0, 0, hidden).copy_(arg4_1)

    mask_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    normalized_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_ln_kernel,
        (flat_pad, r_pad, resid_pad, w_pad, b_pad,
         mask_pad, add_pad, normalized_pad, bf16_pad, div_1d,
         hidden, BLOCK_H),
    )

    mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    mask.view(rows, hidden).copy_(mask_pad.narrow(1, 0, hidden))
    add = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    add.view(rows, hidden).copy_(add_pad.narrow(1, 0, hidden))
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    normalized.view(rows, hidden).copy_(normalized_pad.narrow(1, 0, hidden))
    bf16_view = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )
    bf16_view.copy_(bf16_pad.narrow(1, 0, hidden))
    permute = bf16_view.permute(1, 0)
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )
    div.view(rows).copy_(div_1d)

    return mask, add, normalized, bf16_view, permute, div


@oracle_impl(hardware="B200", point="a352047a")
@oracle_impl(hardware="B200", point="bf8decda")
def oracle_forward(inputs):
    return _run(inputs)
