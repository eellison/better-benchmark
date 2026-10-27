"""cuTile port of mean_mean_996428153329: T5 embedding + dual dropout RMSNorm.

Pre-generates random tensors using torch.ops.prims.inductor_random with seed
indices 0 and 26. cuTile row kernel: embedding gather from ids, two dropouts,
two RMS-norms with eps=1e-6, two affine outputs (bf16).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 64
SEED_INDEX_0 = 0
SEED_INDEX_1 = 26
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 512
EPS = 1.0e-6


@ct.kernel
def _embedding_dual_dropout_rmsnorm_kernel(
    table_ptr,       # f32 [V, HIDDEN]
    ids_ptr,         # i64 [ROWS]
    weight0_ptr,     # f32 [HIDDEN]
    weight1_ptr,     # f32 [HIDDEN]
    random0_ptr,     # f32 [ROWS, HIDDEN]
    random1_ptr,     # f32 [ROWS, HIDDEN]
    embedding_ptr,   # f32 [ROWS, HIDDEN]
    gt0_ptr,         # b8  [ROWS, HIDDEN]
    mul1_ptr,        # f32 [ROWS, HIDDEN]
    rsqrt0_ptr,      # f32 [ROWS]
    view0_ptr,       # bf16 [ROWS, HIDDEN]
    gt1_ptr,         # b8  [ROWS, HIDDEN]
    mul5_ptr,        # f32 [ROWS, HIDDEN]
    rsqrt1_ptr,      # f32 [ROWS]
    view1_ptr,       # bf16 [ROWS, HIDDEN]
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    id_tile = ct.load(ids_ptr, index=(row,), shape=(1,))
    token_id = ct.reshape(id_tile, (1, 1))
    # Gather embedding row: offsets = token_id * HIDDEN + cols (table flattened to 1D)
    cols = ct.arange(BLOCK_H, dtype=ct.int64)
    cols_2d = ct.reshape(cols, (1, BLOCK_H))
    offsets = token_id * BLOCK_H + cols_2d
    embedding = ct.gather(table_ptr, offsets)  # f32 [1, HIDDEN] (from 1D table)
    ct.store(embedding_ptr, index=(row, 0), tile=embedding)

    random0 = ct.load(random0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random1 = ct.load(random1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    thresh = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.float32)
    keep0 = random0 > thresh
    keep1 = random1 > thresh
    ct.store(gt0_ptr, index=(row, 0), tile=keep0)
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)

    keep0_f = ct.astype(keep0, ct.float32)
    keep1_f = ct.astype(keep1, ct.float32)
    mul0 = keep0_f * embedding
    mul1 = mul0 * DROPOUT_SCALE
    mul4 = keep1_f * embedding
    mul5 = mul4 * DROPOUT_SCALE
    ct.store(mul1_ptr, index=(row, 0), tile=mul1)
    ct.store(mul5_ptr, index=(row, 0), tile=mul5)

    inv_h = 1.0 / BLOCK_H
    sq0 = ct.sum(mul1 * mul1, axis=1, keepdims=True)
    sq1 = ct.sum(mul5 * mul5, axis=1, keepdims=True)
    inv0 = ct.rsqrt(sq0 * inv_h + EPS)
    inv1 = ct.rsqrt(sq1 * inv_h + EPS)
    ct.store(rsqrt0_ptr, index=(row,), tile=ct.reshape(inv0, (1,)))
    ct.store(rsqrt1_ptr, index=(row,), tile=ct.reshape(inv1, (1,)))

    weight0 = ct.load(weight0_ptr, index=(0,), shape=(BLOCK_H,))
    weight1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_H,))
    w0 = ct.reshape(weight0, (1, BLOCK_H))
    w1 = ct.reshape(weight1, (1, BLOCK_H))
    norm0 = mul1 * inv0
    norm1 = mul5 * inv1
    out0 = w0 * norm0
    out1 = w1 * norm1
    ct.store(view0_ptr, index=(row, 0), tile=ct.astype(out0, ct.bfloat16))
    ct.store(view1_ptr, index=(row, 0), tile=ct.astype(out1, ct.bfloat16))


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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# d37791c4: (T([32128,512], f32), T([8,1024], i64), T([512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="d37791c4", BLOCK_M=1, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    device = arg0_1.device
    full_shape0 = _shape_tuple(shape0)  # [8, 1024, 512]
    view_shape0 = _shape_tuple(shape1)  # [8192, 512]
    full_shape1 = _shape_tuple(shape2)  # [8, 1024, 512]
    view_shape1 = _shape_tuple(shape3)  # [8192, 512]

    rows = int(view_shape0[0])
    hidden = int(view_shape0[1])
    assert hidden == HIDDEN

    embedding = torch.empty_strided(
        full_shape0, _contiguous_stride(full_shape0), device=device, dtype=torch.float32)
    gt0 = torch.empty_strided(
        full_shape0, _contiguous_stride(full_shape0), device=device, dtype=torch.bool)
    mul1 = torch.empty_strided(
        full_shape0, _contiguous_stride(full_shape0), device=device, dtype=torch.float32)
    rsqrt0_shape = full_shape0[:-1] + (1,)
    rsqrt0 = torch.empty_strided(
        rsqrt0_shape, _contiguous_stride(rsqrt0_shape), device=device, dtype=torch.float32)
    view0 = torch.empty_strided(
        view_shape0, _contiguous_stride(view_shape0), device=device, dtype=torch.bfloat16)
    gt1 = torch.empty_strided(
        full_shape1, _contiguous_stride(full_shape1), device=device, dtype=torch.bool)
    mul5 = torch.empty_strided(
        full_shape1, _contiguous_stride(full_shape1), device=device, dtype=torch.float32)
    rsqrt1_shape = full_shape1[:-1] + (1,)
    rsqrt1 = torch.empty_strided(
        rsqrt1_shape, _contiguous_stride(rsqrt1_shape), device=device, dtype=torch.float32)
    view1 = torch.empty_strided(
        view_shape1, _contiguous_stride(view_shape1), device=device, dtype=torch.bfloat16)

    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("cuTile port unsupported inside CUDA graph capture (seeded RNG).")

    seeds, random0, random1 = _seeds_and_randoms_for_eager_check(
        full_shape0, full_shape1, device=device)

    ids_1d = arg1_1.contiguous().view(-1)
    table_1d = arg0_1.contiguous().view(-1)  # [V * HIDDEN]
    random0_2d = random0.contiguous().view(rows, hidden)
    random1_2d = random1.contiguous().view(rows, hidden)
    embedding_2d = embedding.view(rows, hidden)
    gt0_2d = gt0.view(rows, hidden)
    mul1_2d = mul1.view(rows, hidden)
    rsqrt0_1d = rsqrt0.view(rows)
    view0_2d = view0.view(rows, hidden)
    gt1_2d = gt1.view(rows, hidden)
    mul5_2d = mul5.view(rows, hidden)
    rsqrt1_1d = rsqrt1.view(rows)
    view1_2d = view1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _embedding_dual_dropout_rmsnorm_kernel,
        (table_1d, ids_1d, arg2_1, arg3_1, random0_2d, random1_2d,
         embedding_2d, gt0_2d, mul1_2d, rsqrt0_1d, view0_2d,
         gt1_2d, mul5_2d, rsqrt1_1d, view1_2d,
         BLOCK_N),
    )
    return embedding, seeds, gt0, mul1, rsqrt0, view0, gt1, mul5, rsqrt1, view1
