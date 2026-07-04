"""cuTile port of pointwise_ac8d8e1c9b73: DeepRecommender SELU + seeded dropout.

The Repro generates fresh seeds each forward via `inductor_seeds`, so the mask
outputs are stochastic and skipped by the numerics gate. We still emit a cuTile
kernel that computes the SELU activation and applies a mask/scale from a random
tensor pre-generated via `inductor_random` on the host side.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 1
SEED_INDEX = 0
THRESHOLD = 0.8
SCALE = 5.000000000000001


@ct.kernel
def _selu_dropout_kernel(
    x_ptr,          # bf16 [total]
    random_ptr,     # f32 [total]
    mask_ptr,       # bool [total]
    out_ptr,        # bf16 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    rand_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))

    x_f = ct.astype(x_bf, ct.float32)
    positive = x_f * 1.0507009873554805
    # expm1(x) = exp(x) - 1
    negative = (ct.exp(x_f) - 1.0) * 1.7580993408473766
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    selu_f = ct.where(x_f > zero, positive, negative)
    selu_bf = ct.astype(selu_f, ct.bfloat16)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK,), THRESHOLD, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf

    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    dropped_bf = ct.astype(
        ct.astype(ct.where(keep, selu_bf, zero_bf), ct.float32), ct.bfloat16
    )
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * SCALE, ct.bfloat16)

    ct.store(mask_ptr, index=(pid,), tile=keep)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="0f3e2fa1", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, random_shape_param = inputs
    random_shape = _shape_tuple(random_shape_param)
    device = x.device

    mask = torch.empty_strided(random_shape, _contiguous_stride(random_shape), device=device, dtype=torch.bool)
    out = torch.empty_strided(random_shape, _contiguous_stride(random_shape), device=device, dtype=torch.bfloat16)

    # Match Repro: generate fresh seeds -> random tensor.
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(random_shape, seed, "rand")

    total = x.numel()
    x_flat = x.contiguous().view(total)
    random_flat = random.contiguous().view(total)
    mask_flat = mask.view(total)
    out_flat = out.view(total)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _selu_dropout_kernel,
        (x_flat, random_flat, mask_flat, out_flat, BLOCK),
    )
    return mask, out
