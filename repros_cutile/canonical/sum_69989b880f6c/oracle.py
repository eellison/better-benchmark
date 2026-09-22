"""cuTile port of sum_69989b880f6c: Demucs stochastic augment (all outputs stochastic).

Uses torch for the sort/randint/gather (multi-stage RNG). A single cuTile
kernel reduces the final tensor for sum_1 across dim=1.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_reduce_kernel(
    x_ptr,      # f32 [rows, C, cols]
    out_ptr,    # f32 [rows, cols]
    C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_W: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)
    x = ct.load(x_ptr, index=(row, 0, col_block), shape=(1, BLOCK_C, BLOCK_W))
    total = ct.sum(x, axis=1)  # -> (1, BLOCK_W)
    ct.store(out_ptr, index=(row, col_block), tile=ct.reshape(total, (1, BLOCK_W)))


def _shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="764a0dde", BLOCK_C=4, BLOCK_W=1024)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_W: int):
    arg0_1, shape0, shape1, shape2, shape3, shape4, shape5, shape6, shape7, shape8 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("cuTile port unsupported inside CUDA graph capture (seeded RNG).")

    # Do the stochastic ops in torch
    inductor_seeds = torch.ops.prims.inductor_seeds.default(4, device)
    seed3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
    inductor_random = torch.ops.prims.inductor_random.default(
        _shape(shape0), seed3, "rand"
    )
    sorted_vals, sorted_idx = torch.sort(inductor_random, dim=1)
    slice_1 = arg0_1[:, 1:]

    seed0 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
    randint_0 = torch.ops.prims.inductor_randint.default(0, 2, _shape(shape1), seed0)
    sub_val = randint_0.to(torch.float32) * 2 - 1
    mul_1 = slice_1 * sub_val

    seed1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
    randint_1 = torch.ops.prims.inductor_randint.default(0, 2, _shape(shape2), seed1)
    expand = randint_1.expand(_shape(shape3))
    gather = torch.gather(mul_1, 2, expand)
    sub_1 = 1 - expand
    gather_1 = torch.gather(mul_1, 2, sub_1)
    cat = torch.cat([gather, gather_1], dim=2)

    iota = torch.ops.prims.iota.default(
        382788, start=0, step=1, dtype=torch.int64, device=device, requires_grad=False
    )
    seed2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
    randint_2 = torch.ops.prims.inductor_randint.default(0, 44100, _shape(shape4), seed2)
    expand_1 = randint_2.expand(_shape(shape5))
    add = iota + expand_1
    gather_2 = torch.gather(cat, 3, add)
    view = gather_2.view(_shape(shape6))
    expand_2 = sorted_idx.expand(_shape(shape7))
    gather_3 = torch.gather(view, 1, expand_2)
    view_1 = gather_3.view(_shape(shape8))

    # sum_1 via cuTile
    B, C, D, W = int(view_1.shape[0]), int(view_1.shape[1]), int(view_1.shape[2]), int(view_1.shape[3])
    rows = B  # 4
    cols = D * W  # 2 * 382788
    view_1_contig = view_1.contiguous().view(rows, C, cols)
    sum_1 = torch.empty((rows, cols), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, ct.cdiv(cols, BLOCK_W), 1),
        _row_reduce_kernel,
        (view_1_contig, sum_1, C, BLOCK_C, BLOCK_W),
    )
    sum_1_3d = sum_1.view(rows, D, W)
    return view_1, sum_1_3d
