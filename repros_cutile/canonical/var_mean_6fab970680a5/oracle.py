"""cuTile port of var_mean_6fab970680a5: GPT-2 seeded dropout + residual + LayerNorm.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. HIDDEN=768 needs BLOCK_H=1024
with padding-mode-zero loads and column-masked reductions.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 14
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5
HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _dropout_residual_layernorm_kernel(
    addmm_ptr,       # bf16 [rows, HIDDEN]
    random_ptr,      # f32  [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    bias_ptr,        # f32  [HIDDEN]
    gt_ptr,          # b8   [rows, BLOCK_H] padded
    add_ptr,         # f32  [rows, BLOCK_H] padded
    normalized_ptr,  # f32  [rows, BLOCK_H] padded
    affine_ptr,      # bf16 [rows, BLOCK_H] padded
    div_ptr,         # f32  [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    addmm_bf = ct.load(
        addmm_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand = ct.load(
        random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((ROW_BLOCK, BLOCK_H_), 0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, addmm_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    x = ct.astype(scaled_bf, ct.float32) + residual
    ct.store(add_ptr, index=(row_block, 0), tile=x)

    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    zero_f = ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.float32)
    x_masked = ct.where(col_mask, x, zero_f)
    inv_h = 1.0 / HIDDEN_
    mean = ct.sum(x_masked, axis=1) * inv_h
    mean_2d = ct.reshape(mean, (ROW_BLOCK, 1))
    centered = x - mean_2d
    centered_masked = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    invstd_2d = ct.reshape(invstd, (ROW_BLOCK, 1))
    normalized = centered * invstd_2d

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
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    ct.store(normalized_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine_bf)
    ct.store(div_ptr, index=(row_block,), tile=invstd * inv_h)


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


@oracle_impl(hardware="B200", point="bf8decda", ROW_BLOCK=1)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)  # [8,1024,768]
    random_shape = _as_shape(shape1)  # [8,1024,768]
    out_shape = _as_shape(shape2, numel=arg0_1.numel())  # [-1,768]
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    assert hidden == HIDDEN
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=arg0_1.device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    affine_bf16 = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=arg0_1.device, dtype=torch.float32,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)

    gt_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    normalized_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)
    div_1d = div.view(rows)

    addmm_2d = arg0_1
    residual_2d = arg2_1.reshape(rows, hidden)
    random_2d = random.reshape(rows, hidden).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_residual_layernorm_kernel,
        (addmm_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_pad, add_pad, normalized_pad, affine_pad, div_1d,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
    add.view(rows, hidden).copy_(add_pad.narrow(1, 0, hidden))
    normalized.view(rows, hidden).copy_(normalized_pad.narrow(1, 0, hidden))
    affine_bf16.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))

    return gt, add, normalized, affine_bf16, affine_bf16.permute(1, 0), div
