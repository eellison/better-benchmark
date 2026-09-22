"""cuTile port of var_mean_37ea415cca37: BERT embedding LayerNorm + dropout.

Runs the embedding sums via torch, LayerNorm + dropout via cuTile.
Returns (gather, mul, inductor_seeds, gt, mul_3, view, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
SEED_COUNT = 37
SEED_INDEX = 0


@ct.kernel
def _layernorm_dropout_kernel(
    x_ptr,          # f32 [rows, HIDDEN]
    random_ptr,     # f32 [rows, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    mul_ptr,        # f32 [rows, HIDDEN]
    gt_ptr,         # bool [rows, HIDDEN]
    mul3_ptr,       # f32 [rows, HIDDEN]
    view_ptr,       # bf16 [rows, HIDDEN]
    div_ptr,        # f32 [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_f = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weight_f = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias_f = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(x_f, axis=1, keepdims=True) * inv_h
    centered = x_f - mean_1d
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    rsqrt_val = ct.rsqrt(variance + EPS)
    mul_val = centered * rsqrt_val
    ct.store(mul_ptr, index=(row, 0), tile=mul_val)

    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    add3 = mul_val * weight_2d + bias_2d

    threshold_f = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.float32)
    keep = rand_f > threshold_f
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    dropped = keep_f * add3
    scaled = dropped * DROPOUT_SCALE
    ct.store(mul3_ptr, index=(row, 0), tile=scaled)
    ct.store(view_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(rsqrt_val * inv_h, (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="e57d24c8", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     shape0, shape1, shape2) = inputs
    device = arg0_1.device

    # Reproduce the input transformations via torch (embeddings, etc.)
    expand = arg0_1.expand(1, -1)
    gather = torch.gather(expand, 1, arg1_1)
    expand_1 = gather.expand(*tuple(int(d) for d in shape0))
    embedding = torch.embedding(arg2_1, arg3_1, 0)
    embedding_1 = torch.embedding(arg4_1, expand_1)
    add = embedding + embedding_1
    embedding_2 = torch.embedding(arg5_1, arg1_1)
    add_1 = add + embedding_2

    random_shape = tuple(int(d) for d in shape1)
    out_shape = tuple(int(d) for d in shape2)

    base_shape = add_1.shape
    base_stride = _contiguous_stride(base_shape)
    div_shape = tuple(base_shape[:-1]) + (1,)
    hidden = int(base_shape[-1])
    rows = 1
    for d in base_shape[:-1]:
        rows *= int(d)

    mul_out = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    gt = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    mul3 = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    view = torch.empty_strided(out_shape, _contiguous_stride(out_shape), device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape), device=device, dtype=torch.float32)

    # Fresh seeds each call — stochastic.
    inductor_seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(random_shape, seed, "rand")

    add_1_2d = add_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    mul_2d = mul_out.view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    mul3_2d = mul3.view(rows, hidden)
    view_2d = view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_dropout_kernel,
        (add_1_2d, random_2d, arg6_1, arg7_1,
         mul_2d, gt_2d, mul3_2d, view_2d, div_1d,
         hidden, BLOCK_H),
    )
    return gather, mul_out, inductor_seeds, gt, mul3, view, div
