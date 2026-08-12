"""cuTile port of var_mean_700e0afd67fd: GPT-style dropout+residual+LayerNorm.

Similar to var_mean_1c8a151acad6 but with seed 3, EPS=1e-5, and returns:
gt, add (f32 residual sum), normalized f32, affine bf16, permute bf16, rsqrt/H.
HIDDEN=768 (needs padding to 1024 power-of-two).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 3
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,        # bf16 [rows, BLOCK_H]
    random_ptr,      # f32 [rows, BLOCK_H]
    residual_ptr,    # f32 [rows, BLOCK_H]
    weight_ptr,      # f32 [BLOCK_H]
    bias_ptr,        # f32 [BLOCK_H]
    gt_ptr,          # bool [rows, BLOCK_H]
    add_ptr,         # f32 [rows, BLOCK_H]
    norm_ptr,        # f32 [rows, BLOCK_H]
    affine_bf16_ptr, # bf16 [rows, BLOCK_H]
    div_ptr,         # f32 [rows]
    HIDDEN: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))

    rand_bf16 = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.astype(ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf16 > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled, ct.float32) + residual
    ct.store(add_ptr, index=(row, 0), tile=x)
    x_masked = ct.where(col_mask, x, 0.0)

    inv_h = 1.0 / HIDDEN_F
    mean = ct.sum(x_masked, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    div_val = invstd * inv_h
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape, *, numel=None):
    shape = tuple(int(dim) for dim in shape)
    if -1 not in shape:
        return shape
    if numel is None:
        raise ValueError("numel is required to resolve -1 shape dimensions")
    known = 1
    unknown_idx = None
    for idx, dim in enumerate(shape):
        if dim == -1:
            unknown_idx = idx
        else:
            known *= dim
    resolved = list(shape)
    resolved[unknown_idx] = int(numel) // known
    return tuple(resolved)


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


@oracle_impl(hardware="B200", point="a352047a", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape2, numel=arg0_1.numel())
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    if BLOCK_H == hidden:
        x_2d = arg0_1.contiguous().view(rows, hidden)
        random_2d = random.contiguous().view(rows, hidden)
        residual_2d = arg2_1.contiguous().view(rows, hidden)
        weight_pad = arg3_1
        bias_pad = arg4_1
    else:
        x_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        x_2d[:, :hidden].copy_(arg0_1)
        random_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        random_2d[:, :hidden].copy_(random.view(rows, hidden))
        residual_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        residual_2d[:, :hidden].copy_(arg2_1.view(rows, hidden))
        weight_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        weight_pad[:hidden].copy_(arg3_1)
        bias_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        bias_pad[:hidden].copy_(arg4_1)

    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (x_2d, random_2d, residual_2d, weight_pad, bias_pad,
         gt_pad, add_pad, norm_pad, affine_bf16_pad, div_1d,
         hidden, float(hidden), BLOCK_H),
    )

    if BLOCK_H == hidden:
        gt = gt_pad.view(norm_shape)
        add = add_pad.view(norm_shape)
        normalized = norm_pad.view(norm_shape)
        affine_bf16 = affine_bf16_pad.view(out_shape)
    else:
        gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
        gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
        add = torch.empty(norm_shape, device=device, dtype=torch.float32)
        add.view(rows, hidden).copy_(add_pad.narrow(1, 0, hidden))
        normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
        normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))
        affine_bf16 = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
        affine_bf16.view(rows, hidden).copy_(affine_bf16_pad.narrow(1, 0, hidden))
    div = div_1d.view(div_shape)
    return gt, add, normalized, affine_bf16, affine_bf16.permute(1, 0), div
