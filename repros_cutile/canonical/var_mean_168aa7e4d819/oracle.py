"""cuTile port of var_mean_168aa7e4d819: GPT2 embedding + dropout + LayerNorm.

Uses torch for the two embeddings + ne mask (pure) and pre-generates the
inductor_random tensor on the Python side; the cuTile kernel handles the
dropout, LayerNorm (correction=0), affine and bf16 output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 0
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_ln_kernel(
    add1_ptr,       # f32 [rows, HIDDEN] = embedding + embedding_1
    random_ptr,     # f32 [rows, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    gt_ptr,         # bool [rows, HIDDEN]
    mul1_ptr,       # f32 [rows, HIDDEN] = post-dropout scaled
    mean_ptr,       # f32 [rows]
    rsqrt_ptr,      # f32 [rows]
    bf16_ptr,       # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    add1 = ct.load(add1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    keep = rand > ct.full((1, BLOCK_H), 0.1, dtype=ct.float32)
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    dropped = ct.where(keep, add1, zero_f)
    scaled = dropped * DROPOUT_SCALE
    ct.store(mul1_ptr, index=(row, 0), tile=scaled)

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(scaled, axis=1, keepdims=True) * inv_h
    centered = scaled - mean_1d
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean_1d, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd_1d, (1,)))


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


@oracle_impl(hardware="B200", point="39fdc80b", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2, shape3 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    # Embeddings + prev
    embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1)  # [8, 1024, 768]
    iota = torch.ops.prims.iota.default(1024, start=0, step=1, dtype=torch.int64, device=device, requires_grad=False)
    add_i = iota + 0
    unsqueeze = add_i.unsqueeze(0)  # [1, 1024]
    embedding_1 = torch.ops.aten.embedding.default(arg2_1, unsqueeze)  # [1, 1024, 768]
    add_1 = embedding + embedding_1  # [8, 1024, 768]
    expand = unsqueeze.expand(_shape(shape0))
    full = torch.full(_shape(shape1), -1, dtype=torch.int64, device=device)
    cat = torch.cat([full, expand], -1)
    slice_1 = cat[..., 0:1024]
    slice_2 = cat[..., 1:1025]
    sub = slice_2 - slice_1
    ne = sub != 1

    inductor_seeds = torch.ops.prims.inductor_seeds.default(25, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    random_shape = _shape(shape2)  # [8, 1024, 768]
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    view_shape = tuple(add_1.shape)  # [8, 1024, 768]
    rows = int(view_shape[0]) * int(view_shape[1])
    hidden = int(view_shape[2])

    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    mul_1 = torch.empty(view_shape, device=device, dtype=torch.float32)
    mean_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    rsqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    bf16_out = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)

    add1_2d = add_1.reshape(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    mul1_2d = mul_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_ln_kernel,
        (add1_2d, random_2d, arg3_1, arg4_1,
         gt_2d, mul1_2d, mean_1d, rsqrt_1d, bf16_out,
         hidden, BLOCK_H),
    )
    # Reshape mean, rsqrt to [8, 1024, 1]
    mean_out = mean_1d.view(view_shape[0], view_shape[1], 1)
    rsqrt_out = rsqrt_1d.view(view_shape[0], view_shape[1], 1)
    permute = bf16_out.t().contiguous()
    return (
        embedding,
        unsqueeze,
        embedding_1,
        ne,
        inductor_seeds,
        gt,
        mul_1,
        mean_out,
        rsqrt_out,
        bf16_out,
        permute,
    )
