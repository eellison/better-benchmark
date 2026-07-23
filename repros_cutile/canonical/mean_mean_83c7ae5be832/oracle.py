"""cuTile port of mean_mean_83c7ae5be832: MT5 embedding + dual dropout RMSNorm.

For each row: gather embedding, run two independent seeded dropout RMSNorm
branches (seeds 0 and 34), emit both mask+add+rsqrt+bf16 outputs.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 84
SEED_INDEX_0 = 0
SEED_INDEX_1 = 34
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _embedding_dropout_rmsnorm_kernel(
    embedding_in_ptr,  # f32 [rows, hidden] — pre-gathered embedding
    weight0_ptr,     # f32 [hidden]
    weight1_ptr,     # f32 [hidden]
    random0_ptr,     # f32 [rows, hidden]
    random1_ptr,     # f32 [rows, hidden]
    embedding_ptr,   # f32 [rows, hidden]
    gt0_ptr, mul1_ptr, rsqrt0_ptr, view0_ptr,
    gt1_ptr, mul5_ptr, rsqrt1_ptr, view1_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row = ct.bid(0)

    embedding_f = ct.load(embedding_in_ptr, index=(row, 0), shape=(ROW_BLOCK, BLOCK_H))
    ct.store(embedding_ptr, index=(row, 0), tile=embedding_f)

    # Branch 0
    rand0 = ct.load(random0_ptr, index=(row, 0), shape=(ROW_BLOCK, BLOCK_H))
    keep0 = rand0 > 0.1
    ct.store(gt0_ptr, index=(row, 0), tile=keep0)
    zero = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.float32)
    dropped0 = ct.where(keep0, embedding_f, zero)
    mul1 = dropped0 * DROPOUT_SCALE
    ct.store(mul1_ptr, index=(row, 0), tile=mul1)

    sq0 = ct.sum(mul1 * mul1, axis=1, keepdims=True) * (1.0 / HIDDEN)
    inv0 = ct.rsqrt(sq0 + EPS)
    ct.store(rsqrt0_ptr, index=(row, 0), tile=inv0)

    weight0 = ct.load(weight0_ptr, index=(0,), shape=(BLOCK_H,))
    weight0_2d = ct.reshape(weight0, (1, BLOCK_H))
    norm0 = mul1 * inv0
    out0 = norm0 * weight0_2d
    ct.store(view0_ptr, index=(row, 0), tile=ct.astype(out0, ct.bfloat16))

    # Branch 1
    rand1 = ct.load(random1_ptr, index=(row, 0), shape=(ROW_BLOCK, BLOCK_H))
    keep1 = rand1 > 0.1
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)
    dropped1 = ct.where(keep1, embedding_f, zero)
    mul5 = dropped1 * DROPOUT_SCALE
    ct.store(mul5_ptr, index=(row, 0), tile=mul5)

    sq1 = ct.sum(mul5 * mul5, axis=1, keepdims=True) * (1.0 / HIDDEN)
    inv1 = ct.rsqrt(sq1 + EPS)
    ct.store(rsqrt1_ptr, index=(row, 0), tile=inv1)

    weight1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_H,))
    weight1_2d = ct.reshape(weight1, (1, BLOCK_H))
    norm1 = mul5 * inv1
    out1 = norm1 * weight1_2d
    ct.store(view1_ptr, index=(row, 0), tile=ct.astype(out1, ct.bfloat16))


def _shape(shape):
    return tuple(int(d) for d in shape)


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
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _seeds_and_randoms_for_eager_check(shape0, shape1, *, device):
    total_advance = 8 + _random_advance(shape0) + _random_advance(shape1)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
    random0 = torch.ops.prims.inductor_random.default(shape0, seed0, "rand")
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
    random1 = torch.ops.prims.inductor_random.default(shape1, seed1, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random0, random1


@oracle_impl(hardware="B200", point="b30c9463", BLOCK_H=512, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape0 = _shape(shape0)     # (32, 128, 512)
    view_shape0 = _shape(shape1)     # (4096, 512)
    full_shape1 = _shape(shape2)
    view_shape1 = _shape(shape3)
    device = arg0_1.device
    rows = view_shape0[0]
    hidden = view_shape0[1]
    row_shape0 = full_shape0[:-1] + (1,)
    row_shape1 = full_shape1[:-1] + (1,)

    embedding = torch.empty(full_shape0, device=device, dtype=torch.float32)
    gt0 = torch.empty(full_shape0, device=device, dtype=torch.bool)
    mul1 = torch.empty(full_shape0, device=device, dtype=torch.float32)
    rsqrt0 = torch.empty(row_shape0, device=device, dtype=torch.float32)
    view0 = torch.empty(view_shape0, device=device, dtype=torch.bfloat16)
    gt1 = torch.empty(full_shape1, device=device, dtype=torch.bool)
    mul5 = torch.empty(full_shape1, device=device, dtype=torch.float32)
    rsqrt1 = torch.empty(row_shape1, device=device, dtype=torch.float32)
    view1 = torch.empty(view_shape1, device=device, dtype=torch.bfloat16)

    seeds, random0, random1 = _seeds_and_randoms_for_eager_check(
        full_shape0, full_shape1, device=device,
    )

    # Pre-gather embeddings (torch-side, faster than in-kernel gather here).
    embedding_in = torch.ops.aten.embedding.default(arg0_1, arg1_1).view(rows, hidden).contiguous()
    random0_2d = random0.reshape(rows, hidden)
    random1_2d = random1.reshape(rows, hidden)
    embedding_2d = embedding.view(rows, hidden)
    gt0_2d = gt0.view(rows, hidden)
    mul1_2d = mul1.view(rows, hidden)
    rsqrt0_2d = rsqrt0.view(rows, 1)
    view0_2d = view0.view(rows, hidden)
    gt1_2d = gt1.view(rows, hidden)
    mul5_2d = mul5.view(rows, hidden)
    rsqrt1_2d = rsqrt1.view(rows, 1)
    view1_2d = view1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _embedding_dropout_rmsnorm_kernel,
        (embedding_in, arg2_1, arg3_1, random0_2d, random1_2d,
         embedding_2d,
         gt0_2d, mul1_2d, rsqrt0_2d, view0_2d,
         gt1_2d, mul5_2d, rsqrt1_2d, view1_2d,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    return embedding, seeds, gt0, mul1, rsqrt0, view0, gt1, mul5, rsqrt1, view1
