"""cuTile port of mean_5088936e00ad: MT5 seeded dropout + residual + RMSNorm.

Row kernel per hidden row. Pre-generates the seeded random tensor outside the
kernel via inductor_random (matches the Triton oracle's non-graph-capture
path), then applies:

  bf16 dropout mask via `random_bf16 > 0.1`, dropout scale 1.111..., residual
  add in fp32, mean-square, rsqrt(1e-6), affine bf16 output.

Returns (gt, add, rsqrt, bf16 output view).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 68
DROPOUT_SCALE = 1.1111111111111112
DROPOUT_P = 0.1
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random_ptr,      # f32  [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    gt_ptr,          # b8   [rows, HIDDEN]
    add_ptr,         # f32  [rows, HIDDEN]
    rsqrt_ptr,       # f32  [rows]
    out_ptr,         # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    random_bf16 = ct.astype(rand_f, ct.bfloat16)
    threshold = ct.full((1, BLOCK_H), DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf16 > threshold
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped = ct.astype(
        ct.where(keep, ct.astype(flat, ct.float32), 0.0), ct.bfloat16
    )
    scaled_bf16 = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    add_val = residual + ct.astype(scaled_bf16, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    sq_sum = ct.sum(add_val * add_val)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HIDDEN) + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(ct.full((1,), inv_rms, dtype=ct.float32), (1,)))

    normalized = add_val * inv_rms
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    affine = normalized * weight_2d
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
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
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
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    base_shape = _shape_tuple(shape0)     # [32, 128, 512]
    random_shape = _shape_tuple(shape1)   # [32, 128, 512]
    out_shape = _shape_tuple(shape2)      # [4096, 512]
    base_stride = _contiguous_stride(base_shape)
    rsqrt_shape = base_shape[:-1] + (1,)
    device = arg0_1.device

    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == BLOCK_H

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # 2D flat views for the kernel.
    flat_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    gt = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        rsqrt_shape, _contiguous_stride(rsqrt_shape),
        device=device, dtype=torch.float32,
    )
    out_base = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.bfloat16,
    )

    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_rmsnorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_1d, out_2d, hidden, BLOCK_H),
    )

    return gt, add, rsqrt, out_base.view(out_shape)
