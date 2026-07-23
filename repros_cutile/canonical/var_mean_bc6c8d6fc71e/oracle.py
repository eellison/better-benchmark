"""cuTile port of var_mean_bc6c8d6fc71e: internally-seeded dropout-residual LayerNorm.

Uses torch.ops.prims.inductor_seeds to generate the seed pack, then
inductor_random to pre-generate the dropout mask; passes it into a
cuTile row kernel.
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


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _seeds_and_random(shape, *, device):
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
    return seeds, random


@ct.kernel
def _dropout_residual_ln_kernel(
    src_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    mask_ptr, add_ptr, mean_ptr, rsqrt_ptr, out_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)
    src = ct.load(src_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                  padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    rand_f32 = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)

    rand_bf16 = ct.astype(rand_f32, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, HIDDEN_PAD), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf16 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, src, zero_bf16)
    scaled_bf16 = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled_bf16, ct.float32) + residual

    zero_f32 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x, zero_f32)
    ct.store(add_ptr, index=(row, 0), tile=x_masked)

    mean_val = ct.sum(x_masked) * (1.0 / HIDDEN_F)
    centered = x - mean_val
    centered_masked = ct.where(valid_2d, centered, zero_f32)
    variance_val = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_F)
    invstd_val = ct.rsqrt(variance_val + EPS)
    normalized = centered * invstd_val

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PAD))
    affine = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean_val, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd_val, (1,)))
    ct.store(out_ptr, index=(row, 0),
             tile=ct.where(valid_2d, ct.astype(affine, ct.bfloat16), zero_bf16))


@oracle_impl(hardware="B200", point="7f3761d4")
@oracle_impl(hardware="B200", point="b3d1cca6")
@oracle_impl(hardware="B200", point="a47484d2")
@oracle_impl(hardware="B200", point="167c73ce")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    batch = int(norm_shape[0])
    tokens = int(norm_shape[1])
    stat_shape = (batch, tokens, 1)

    device = arg0_1.device
    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                             device=device, dtype=torch.bool)
    add = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                              device=device, dtype=torch.float32)
    mean = torch.empty_strided(stat_shape, (tokens, 1, 1),
                               device=device, dtype=torch.float32)
    rsqrt = torch.empty_strided(stat_shape, (tokens, 1, 1),
                                device=device, dtype=torch.float32)
    out = torch.empty_strided(flat_shape, (hidden, 1),
                              device=device, dtype=torch.bfloat16)

    seeds, random = _seeds_and_random(norm_shape, device=device)

    HIDDEN_PAD = _next_pow2(hidden)
    random_2d = random.reshape(rows, hidden)
    residual_2d = arg1_1.reshape(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    mean_1d = mean.view(rows)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (arg0_1, random_2d, residual_2d, arg2_1, arg3_1,
         gt_2d, add_2d, mean_1d, rsqrt_1d, out_2d,
         hidden, HIDDEN_PAD, float(hidden)),
    )
    return seeds, gt, add, mean, rsqrt, out
