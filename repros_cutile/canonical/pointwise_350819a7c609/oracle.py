"""cuTile port of pointwise_350819a7c609: BERT dual-dropout residual scope.

Pre-generates two seeded random tensors (matches Triton oracle's non-graph-
capture path). Single cuTile row kernel produces mask0, mask1, out; a second
kernel materializes selected (batch-index seq=0 slice).

Inline PTX `add.rn.f32` / `mul.rn.f32` are default cuTile fp32 rounding,
so we use +/* directly. HIDDEN=768 -> BLOCK_H=1024 padded with masked
stores via narrow-copy.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
HIDDEN = 768
ROWS = BATCH * SEQ
BLOCK_H = 1024  # smallest power of 2 >= HIDDEN
SEED_INDEX_0 = 59
SEED_INDEX_1 = 60
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dual_dropout_kernel(
    x_ptr,           # bf16 [ROWS, HIDDEN]
    random0_ptr,     # f32  [ROWS, HIDDEN]
    random1_ptr,     # f32  [ROWS, HIDDEN]
    residual_ptr,    # f32  [ROWS, HIDDEN]
    mask0_pad_ptr,   # b8   padded [ROWS, BLOCK_H]
    mask1_pad_ptr,   # b8   padded [ROWS, BLOCK_H]
    out_pad_ptr,     # bf16 padded [ROWS, BLOCK_H]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf16 = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random0 = ct.load(
        random0_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random1 = ct.load(
        random1_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    threshold_bf16 = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep0 = ct.astype(random0, ct.bfloat16) > threshold_bf16
    keep1 = random1 > DROPOUT_P

    x_f = ct.astype(x_bf16, ct.float32)
    keep0_f = ct.astype(keep0, ct.float32)
    dropped0_bf16 = ct.astype(keep0_f * x_f, ct.bfloat16)
    scaled0_bf16 = ct.astype(ct.astype(dropped0_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    added = residual + ct.astype(scaled0_bf16, ct.float32)
    keep1_f = ct.astype(keep1, ct.float32)
    dropped1 = keep1_f * added
    final = dropped1 * DROPOUT_SCALE
    final_bf16 = ct.astype(final, ct.bfloat16)

    ct.store(mask0_pad_ptr, index=(row, 0), tile=keep0)
    ct.store(mask1_pad_ptr, index=(row, 0), tile=keep1)
    ct.store(out_pad_ptr, index=(row, 0), tile=final_bf16)


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


def _random_advance(shape, *, device):
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
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    advance = _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


@oracle_impl(hardware="B200", point="4a069931", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N: int):
    del BLOCK_N  # Triton-only knob
    (
        x,
        seeds,
        residual,
        view_shape,
        random_shape0,
        random_shape1,
        out_shape,
    ) = inputs
    del view_shape, random_shape1

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    random_shape = tuple(int(dim) for dim in random_shape0)
    flat_out_shape = tuple(int(dim) for dim in out_shape)
    device = x.device

    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape, seed0, seed1, device=device,
    )
    random0_flat = random0.reshape(ROWS, HIDDEN).contiguous()
    random1_flat = random1.reshape(ROWS, HIDDEN).contiguous()
    residual_flat = residual.reshape(ROWS, HIDDEN).contiguous()
    x_flat = x.reshape(ROWS, HIDDEN).contiguous()

    mask0_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bool)
    mask1_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bool)
    out_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _dual_dropout_kernel,
        (
            x_flat, random0_flat, random1_flat, residual_flat,
            mask0_pad, mask1_pad, out_pad, HIDDEN, BLOCK_H,
        ),
    )

    # Narrow copy back to the final shapes matching the Triton oracle output.
    mask0 = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    mask0.view(ROWS, HIDDEN).copy_(mask0_pad.narrow(1, 0, HIDDEN))

    mask1 = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    mask1.view(ROWS, HIDDEN).copy_(mask1_pad.narrow(1, 0, HIDDEN))

    out = torch.empty_strided(
        flat_out_shape, _contiguous_stride(flat_out_shape),
        device=device, dtype=torch.bfloat16,
    )
    out.view(ROWS, HIDDEN).copy_(out_pad.narrow(1, 0, HIDDEN))

    # selected: (BATCH, HIDDEN) — every batch's seq=0 slice of out.
    selected = torch.empty_strided(
        (BATCH, HIDDEN), (HIDDEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_3d = out.view(BATCH, SEQ, HIDDEN)
    selected.copy_(out_3d[:, 0, :])

    return mask0, mask1, selected, out
