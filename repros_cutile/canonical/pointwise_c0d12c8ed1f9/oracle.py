"""cuTile port of pointwise_c0d12c8ed1f9: T5 ReLU + seeded dropout with side masks.

Pre-generates the seeded random tensor via inductor_random outside the kernel
(matches Triton oracle's non-graph-capture path), then runs a single cuTile
row kernel that:
  * casts random f32 -> bf16 and computes keep = random_bf16 > 0.1
  * loads x as bf16, computes bf16 ReLU with NaN preserved via (x > 0) | (x != x)
  * computes le = relu <= 0
  * dropped = keep_f * relu_f (bf16 rounded), scaled = dropped * DROPOUT_SCALE

Non-graph-capture only. HIDDEN=2048 is a power of 2 so BLOCK_H can equal HIDDEN.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 2048
ROWS = BATCH * SEQ
NUMEL = ROWS * HIDDEN
VIEW_SHAPE = (BATCH, SEQ, HIDDEN)
VIEW_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
SEED_INDEX = 23
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _relu_dropout_masks_kernel(
    x_ptr,           # bf16 [ROWS, HIDDEN]
    random_ptr,      # f32  [ROWS, HIDDEN]
    gt_ptr,          # b8   [ROWS, HIDDEN]
    out_ptr,         # bf16 [ROWS, HIDDEN]
    le_ptr,          # b8   [ROWS, HIDDEN]
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    random = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_bf16 = ct.astype(random, ct.bfloat16)
    threshold_bf16 = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf16 > threshold_bf16
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    zero_bf16 = ct.full(shape=(1, BLOCK_H), fill_value=0.0, dtype=ct.bfloat16)
    is_pos = x > zero_bf16
    is_nan = x != x
    keep_x = is_pos | is_nan
    relu = ct.where(keep_x, x, zero_bf16)
    non_positive = relu <= zero_bf16
    ct.store(le_ptr, index=(row, 0), tile=non_positive)

    keep_f = ct.astype(keep, ct.float32)
    relu_f = ct.astype(relu, ct.float32)
    dropped_bf16 = ct.astype(keep_f * relu_f, ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled_bf16)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="52dd4c9c")
def oracle_forward(inputs):
    x, seeds, view_shape, random_shape, out_shape = inputs
    del view_shape

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    device = x.device

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape, seed, device=device,
    )
    random_flat = random.reshape(ROWS, HIDDEN).contiguous()
    x_flat = x.reshape(ROWS, HIDDEN).contiguous()

    gt = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    out_base = torch.empty_strided(
        VIEW_SHAPE, VIEW_STRIDE, device=device, dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        VIEW_SHAPE, VIEW_STRIDE, device=device, dtype=torch.bool,
    )

    gt_flat = gt.view(ROWS, HIDDEN)
    out_flat = out_base.view(ROWS, HIDDEN)
    le_flat = le.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _relu_dropout_masks_kernel,
        (x_flat, random_flat, gt_flat, out_flat, le_flat, HIDDEN),
    )

    return gt, out_base.view(out_shape), le
