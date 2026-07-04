"""cuTile port of pointwise_de64dd044945: SqueezeNet ReLU + concat + dropout(p=0.5).

Applies ReLU to two bf16 inputs (in-place-safe by computing masks first),
concatenates, applies fresh-seeded Inductor dropout with p=0.5, scale=2.0,
and emits the two backward masks. cuTile kernel handles the dropout epilogue;
concat/relu backward masks use torch aten ops which are graph-capturable.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0
BLOCK_N = 1024


@ct.kernel
def _relu_concat_dropout_kernel(
    cat_ptr,         # bf16 [total]
    random_ptr,      # f32  [total]
    gt_ptr,          # b8   [total]
    out_ptr,         # bf16 [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    cat_bf = ct.load(cat_ptr, index=(pid,), shape=(BLOCK,))
    rand_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(BLOCK,), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(pid,), tile=keep)

    zero_bf = ct.astype(
        ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32),
        ct.bfloat16,
    )
    dropped = ct.where(keep, cat_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(pid,), tile=scaled_bf)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="30725500")
def oracle_forward(inputs):
    arg0_1, arg1_1, out_shape = inputs
    device = arg0_1.device
    out_shape_t = _as_shape(out_shape)

    # ReLU
    relu0 = torch.relu(arg0_1)
    relu1 = torch.relu(arg1_1)
    cat = torch.cat([relu0, relu1], dim=1)
    n = cat.numel()

    # Fresh seed (Inductor path)
    seeds = torch.ops.prims.inductor_seeds.default(1, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
    random = torch.ops.prims.inductor_random.default(out_shape_t, seed, "rand")
    random_flat = random.contiguous().view(-1)
    cat_flat = cat.contiguous().view(-1)

    gt_flat = torch.empty((n,), device=device, dtype=torch.bool)
    out_flat = torch.empty((n,), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    assert n % BLOCK_N == 0
    ct.launch(
        stream, (n // BLOCK_N, 1, 1), _relu_concat_dropout_kernel,
        (cat_flat, random_flat, gt_flat, out_flat, BLOCK_N),
    )

    gt = torch.empty_strided(
        out_shape_t, _contiguous_stride(out_shape_t),
        device=device, dtype=torch.bool,
    )
    gt.view(-1).copy_(gt_flat)
    out = torch.empty_strided(
        out_shape_t, _contiguous_stride(out_shape_t),
        device=device, dtype=torch.bfloat16,
    )
    out.view(-1).copy_(out_flat)

    le = relu1 <= 0
    le_1 = relu0 <= 0
    return gt, out, le, le_1
