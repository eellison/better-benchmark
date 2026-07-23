"""cuTile port of var_mean_0d6f1eb6e0c6: M2M100 internally-seeded dropout+LN."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 4
SEED_INDEX = 0
BATCH = 64
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 1024
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


def _seeds_and_random(shape, *, device):
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
    return seeds, random


@ct.kernel
def _dropout_ln_kernel(
    src_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, add_ptr, mean_ptr, rsqrt_ptr, out_ptr,
    HIDDEN_: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)
    src = ct.load(src_ptr, index=(row, 0), shape=(1, HIDDEN_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_))
    rand_f32 = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_))

    rand_bf16 = ct.astype(rand_f32, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, HIDDEN_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf16 = ct.full(shape=(1, HIDDEN_), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, src, zero_bf16)
    scaled_bf16 = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled_bf16, ct.float32) + residual
    ct.store(add_ptr, index=(row, 0), tile=x)

    mean_val = ct.sum(x) * (1.0 / HIDDEN_F)
    centered = x - mean_val
    variance_val = ct.sum(centered * centered) * (1.0 / HIDDEN_F)
    invstd_val = ct.rsqrt(variance_val + EPS)
    normalized = centered * invstd_val

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_,))
    weight_2d = ct.reshape(weight, (1, HIDDEN_))
    bias_2d = ct.reshape(bias, (1, HIDDEN_))
    affine = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean_val, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd_val, (1,)))
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="c414de20")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, random_shape, _shape2 = inputs
    base_shape = tuple(int(d) for d in shape0)
    rand_shape = tuple(int(d) for d in random_shape)
    device = arg0_1.device

    gt = torch.empty(base_shape, device=device, dtype=torch.bool)
    add = torch.empty(base_shape, device=device, dtype=torch.float32)
    mean = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    rsqrt = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)

    seeds, random = _seeds_and_random(rand_shape, device=device)

    random_2d = random.view(ROWS, HIDDEN)
    residual_2d = arg1_1.view(ROWS, HIDDEN)
    gt_2d = gt.view(ROWS, HIDDEN)
    add_2d = add.view(ROWS, HIDDEN)
    mean_1d = mean.view(ROWS)
    rsqrt_1d = rsqrt.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _dropout_ln_kernel,
        (arg0_1, random_2d, residual_2d, arg2_1, arg3_1,
         gt_2d, add_2d, mean_1d, rsqrt_1d, out,
         HIDDEN, float(HIDDEN)),
    )
    return seeds, gt, add, mean, rsqrt, out
