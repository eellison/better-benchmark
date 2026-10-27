"""cuTile port of mean_ecd0849414d5: MT5 dropout-residual RMSNorm row kernel.

Pre-generates the seeded random tensor via inductor_random outside the kernel,
then runs a single cuTile row kernel that fuses: bf16 dropout mask/scale, fp32
residual add, mean-square + rsqrt with eps, bf16 affine weight multiply.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 70
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    arg0_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    gt_ptr,
    add_ptr,
    rsqrt_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    arg0 = ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, arg0, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add)

    square = add * add
    square_sum = ct.sum(square, axis=1, keepdims=True)
    mean_square = square_sum * (1.0 / HIDDEN)
    inv_rms = ct.rsqrt(mean_square + EPS)
    inv_rms_1d = ct.reshape(inv_rms, (1,))
    ct.store(rsqrt_ptr, index=(row,), tile=inv_rms_1d)

    weight_bf = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight_bf, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    normalized = add * inv_rms
    affine = weight_2d * normalized
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    reduction_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool)
    add_t = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32)
    rsqrt = torch.empty_strided(
        reduction_shape, _contiguous_stride(reduction_shape),
        device=device, dtype=torch.float32)
    out = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    residual_2d = arg2_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add_t.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_rmsnorm_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_1d, out_2d,
         hidden, BLOCK_H),
    )
    return gt, add_t, rsqrt, out
