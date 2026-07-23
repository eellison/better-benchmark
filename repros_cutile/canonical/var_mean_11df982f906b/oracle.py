"""cuTile port of var_mean_11df982f906b: dropout-residual LayerNorm row kernel.

Pre-generates the seeded random tensor via torch.ops.prims.inductor_random
and passes it as an extra input; the row kernel does dropout mask, residual
add, var/mean, rsqrt, affine, and emits (gt, normalized, affine, bf16_view,
invstd/H) side outputs.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 15
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


@ct.kernel
def _dropout_residual_layernorm_kernel(
    addmm_ptr,       # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    gt_ptr,          # bool [rows, HIDDEN]
    norm_ptr,        # f32 [rows, HIDDEN]
    affine_ptr,      # f32 [rows, HIDDEN]
    bf16_ptr,        # bf16 [rows, HIDDEN]
    div_ptr,         # f32 [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    addmm = ct.load(
        addmm_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(
        ct.where(keep, ct.astype(addmm, ct.float32), 0.0), ct.bfloat16,
    )
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    x = ct.astype(scaled_bf, ct.float32) + residual

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    x_masked = ct.where(col_mask, x, 0.0)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN, (1,)))


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
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
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


# Note: point d9ecc504 has HIDDEN=256, so we use BLOCK_H=256.
# For HIDDEN=768 -> BLOCK_H=1024, HIDDEN=1536 -> BLOCK_H=2048.
@oracle_impl(hardware="B200", point="55aa5fd0", BLOCK_H=2048)
@oracle_impl(hardware="B200", point="243d7832", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="d9ecc504", BLOCK_H=256)
@oracle_impl(hardware="B200", point="d429ff7b", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    affine = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, normalized_2d, affine_2d, bf16_2d, div_1d,
         hidden, BLOCK_H),
    )
    return (gt, normalized, affine, bf16_view, div)
