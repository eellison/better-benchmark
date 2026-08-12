"""cuTile port of var_mean_dc4abd356145: Bart/PLBart/TrOCR dropout + residual + LayerNorm.

Uses `inductor_seeds` + `inductor_random` on the Python side to pre-generate
the random tensor, then runs a single cuTile row kernel that: applies dropout
mask, adds residual (f32), performs LayerNorm (eps=1e-5), affine, and emits
seeds, gt, normalized f32, affine f32, bf16 view, div=invstd/HIDDEN.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 2
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_ln_kernel(
    src_ptr,         # bf16 (rows, HIDDEN)
    random_ptr,      # f32 (rows, HIDDEN)
    residual_ptr,    # f32 (rows, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    bias_ptr,        # f32 (HIDDEN,)
    gt_ptr,          # b8 (rows, HIDDEN)
    normalized_ptr,  # f32 (rows, HIDDEN)
    affine_ptr,      # f32 (rows, HIDDEN)
    bf16_ptr,        # bf16 (rows, HIDDEN)
    div_ptr,         # f32 (rows,)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    src = ct.load(src_ptr, index=(row, 0), shape=(1, BLOCK_H),
                  padding_mode=ct.PaddingMode.ZERO)
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    keep = random_f > DROPOUT_P
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    src_f = ct.astype(src, ct.float32)
    dropped = ct.where(keep, src_f, 0.0) * DROPOUT_SCALE
    add_val = dropped + residual

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    x_masked = ct.where(col_mask, add_val, zero_f)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = add_val - mean
    centered_masked = ct.where(col_mask, centered, zero_f)
    var = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(var + EPS)
    normalized = centered * invstd
    ct.store(normalized_ptr, index=(row, 0), tile=normalized)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                             padding_mode=ct.PaddingMode.ZERO), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN), (1,)))


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


def _random_advance(shape):
    numel = 1
    for d in shape:
        numel *= int(d)
    return (numel + 131071) // 131072


def _seeds_and_random_for_eager_check(shape, *, device):
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        return seeds, random
    total_advance = 8 + _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return seeds, random
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
    return seeds, random


@oracle_impl(hardware="B200", point="a47484d2", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="02ced5d2", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="042c38d1", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = norm_shape[:-1] + (1,)

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool)
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32)
    affine = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=device, dtype=torch.float32)

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
    random_2d = random.view(rows, hidden)
    resid_2d = arg1_1.view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    norm_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (arg0_1, random_2d, resid_2d, arg2_1, arg3_1,
         gt_2d, norm_2d, affine_2d, bf16_2d, div_1d,
         hidden, BLOCK_H),
    )
    return seeds, gt, normalized, affine, bf16_view, div
