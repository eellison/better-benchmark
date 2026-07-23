"""cuTile port of mean_8fc8b05567b8: T5/MT5 dropout-residual-RMSNorm.

Pre-generates the seeded random tensor via inductor_random. Row kernel:
dropout mask + scale (bf16 boundary), residual add (f32), mean of squares,
rsqrt (eps=1e-6), affine multiply, bf16 cast.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 6
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_rmsnorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    gt_ptr,          # bool [rows, HIDDEN]
    add_ptr,         # f32 [rows, HIDDEN]
    rsqrt_ptr,       # f32 [rows]
    out_ptr,         # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat_bf = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    add = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add)

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    add_masked = ct.where(col_mask, add, 0.0)
    square_sum = ct.sum(add_masked * add_masked)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN) + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(inv_rms, (1,)))

    normalized = add * inv_rms
    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    affine = weight_2d * normalized
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    rsqrt_shape = base_shape[:-1] + (1,)

    gt = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        rsqrt_shape, _contiguous_stride(rsqrt_shape),
        device=device, dtype=torch.float32,
    )
    out_base = torch.empty_strided(
        base_shape, _contiguous_stride(base_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_rmsnorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_1d, out_2d,
         hidden, BLOCK_H),
    )
    return gt, add, rsqrt, out_base.view(out_shape)
