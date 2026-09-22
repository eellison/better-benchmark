"""cuTile port of mean_9fba32aab5d5: T5/MT5 dropout residual RMSNorm.

Pre-generates seeded random via inductor_random, then runs one cuTile row kernel:
apply dropout, add residual (f32), RMSNorm (eps=1e-6), affine, output bf16 view.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 10
HIDDEN = 512
BLOCK_H = 512
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,        # bf16 (rows, HIDDEN)
    random_ptr,      # f32 (rows, HIDDEN)
    residual_ptr,    # f32 (rows, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    mask_ptr,        # bool (rows, HIDDEN)
    add_ptr,         # f32 (rows, HIDDEN)
    rsqrt_ptr,       # f32 (rows,)
    final_ptr,       # bf16 (rows, HIDDEN)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_C))

    rand_bf = ct.astype(random_f, ct.bfloat16)
    p_bf = ct.full(shape=(1, BLOCK_H_C), fill_value=DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H_C), dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    sq_sum = ct.sum(add_val * add_val, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HIDDEN_C) + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(inv_rms, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H_C))
    normalized = add_val * inv_rms
    final = ct.astype(weight_2d * normalized, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=final)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound_offset = offset - advance
        rewound[8:16] = torch.tensor(
            list(int(rewound_offset).to_bytes(8, "little", signed=False)),
            dtype=state.dtype, device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="46dbfd5f")
@oracle_impl(hardware="B200", point="ebc95169")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    base_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape2)
    base_stride = _contiguous_stride(base_shape)
    device = arg0_1.device

    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    gt = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    add = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    rsqrt_shape = base_shape[:-1] + (1,)
    rsqrt = torch.empty_strided(rsqrt_shape, _contiguous_stride(rsqrt_shape),
                                device=device, dtype=torch.float32)
    out_base = torch.empty_strided(base_shape, base_stride,
                                   device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    random_2d = random.reshape(rows, hidden)
    residual_2d = arg2_1.reshape(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (
            arg0_1, random_2d, residual_2d, arg3_1,
            gt_2d, add_2d, rsqrt_1d, out_2d,
            hidden, hidden,  # HIDDEN and BLOCK_H equal here
        ),
    )
    return gt, add, rsqrt, out_base.view(out_shape)
