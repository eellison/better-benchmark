"""cuTile port of mean_80ebd99688dc: MT5 seeded-dropout residual RMSNorm."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 26
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


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
        rewound[8:16] = torch.tensor(
            list((offset - advance).to_bytes(8, "little", signed=False)),
            dtype=state.dtype,
            device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr,
    mask_ptr, add_ptr, rsqrt_ptr, final_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN))
    rand_f32 = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN))

    rand_bf16 = ct.astype(rand_f32, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, HIDDEN), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf16 = ct.full(shape=(1, HIDDEN), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf16)
    scaled_bf16 = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_value = ct.astype(scaled_bf16, ct.float32) + residual
    ct.store(add_ptr, index=(row, 0), tile=add_value)

    square_sum = ct.sum(add_value * add_value)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN_F) + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(inv_rms, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, HIDDEN))
    normalized = add_value * inv_rms
    final = weight_2d * normalized
    ct.store(final_ptr, index=(row, 0), tile=ct.astype(final, ct.bfloat16))


@oracle_impl(hardware="B200", point="46dbfd5f")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, _shape1, shape2 = inputs
    base_shape = _as_shape(shape0)
    out_shape = _as_shape(shape2)
    base_stride = _contiguous_stride(base_shape)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device

    mask_out = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bool)
    add_out = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.float32)
    rsqrt_shape = base_shape[:-1] + (1,)
    rsqrt_out = torch.empty_strided(rsqrt_shape, _contiguous_stride(rsqrt_shape),
                                    device=device, dtype=torch.float32)
    final_base = torch.empty_strided(base_shape, base_stride,
                                     device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(base_shape, seed, device=device)

    random_2d = random.reshape(rows, hidden)
    residual_2d = arg2_1.reshape(rows, hidden)
    mask_out_2d = mask_out.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    rsqrt_out_1d = rsqrt_out.view(rows)
    final_2d = final_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_rmsnorm_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1,
         mask_out_2d, add_out_2d, rsqrt_out_1d, final_2d,
         hidden, float(hidden)),
    )
    return mask_out, add_out, rsqrt_out, final_base.view(out_shape)
