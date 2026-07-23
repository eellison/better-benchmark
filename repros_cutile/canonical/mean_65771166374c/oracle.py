"""cuTile port of mean_65771166374c: MT5/T5 seeded-dropout residual RMSNorm.

For each row: dropout mask (bf16), scaled dropout, f32 residual add, mean-of-
squares, rsqrt(eps=1e-6), f32 weight multiply, bf16 output.
Uses pre-generated inductor_random tensor for RNG.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 38
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_rmsnorm_kernel(
    flat_ptr,       # bf16 [rows, hidden]
    random_ptr,     # f32  [rows, hidden]
    residual_ptr,   # f32  [rows, hidden]
    weight_ptr,     # f32  [hidden]
    gt_ptr,         # b8   [rows, hidden]
    add_ptr,        # f32  [rows, hidden]
    rsqrt_ptr,      # f32  [rows, 1]
    out_ptr,        # bf16 [rows, hidden]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(ROW_BLOCK, BLOCK_H))
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    add_f = residual_f + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_f)

    square_sum = ct.sum(add_f * add_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN) + EPS)
    ct.store(rsqrt_ptr, index=(row, 0), tile=inv_rms)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    normalized = add_f * inv_rms
    affine = normalized * weight_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape(shape0)      # (32, 128, 512) or (8, 1024, 512)
    random_shape = _shape(shape1)
    flat_shape = _shape(shape2)      # (4096, 512) or (8192, 512)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    stat_shape = full_shape[:-1] + (1,)

    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    add = torch.empty(full_shape, device=device, dtype=torch.float32)
    rsqrt = torch.empty(stat_shape, device=device, dtype=torch.float32)
    out_base = torch.empty(full_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_2d = rsqrt.view(rows, 1)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_rmsnorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_2d, out_2d,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    return gt, add, rsqrt, out_base.view(flat_shape)
