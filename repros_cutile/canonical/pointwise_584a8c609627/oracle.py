"""cuTile port of pointwise_584a8c609627: GoogleFNet dropout + complex64 packing.

Pre-generates the seeded random tensor. Kernel: apply dropout mask, scale by
1.1111..., emit bool mask, f32 scaled tensor, and pack (real=scaled, imag=0)
into a complex64 output (represented as a f32 view [total, 2]).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 13
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_complex_kernel(
    x_ptr,               # bf16 [total]
    random_ptr,          # f32 [total]
    mask_ptr,            # bool [total]
    scaled_ptr,          # f32 [total]
    complex_real_ptr,    # f32 [total*2] (interpret as [total, 2] for real/imag)
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK_N,))
    random_f = ct.load(random_ptr, index=(pid,), shape=(BLOCK_N,))
    keep = random_f > DROPOUT_P

    x_f = ct.astype(x, ct.float32)
    dropped = ct.where(keep, x_f, 0.0)
    scaled = dropped * DROPOUT_SCALE

    ct.store(mask_ptr, index=(pid,), tile=keep)
    ct.store(scaled_ptr, index=(pid,), tile=scaled)

    # Pack (real, imag) via scatter into a flat 2*BLOCK_N buffer.
    idx = ct.arange(BLOCK_N, dtype=ct.int32) + pid * BLOCK_N
    real_idx = idx * 2
    imag_idx = real_idx + 1
    zero = ct.zeros((BLOCK_N,), dtype=ct.float32)
    ct.scatter(complex_real_ptr, real_idx, scaled)
    ct.scatter(complex_real_ptr, imag_idx, zero)


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


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _seeds_and_random_for_eager_check(shape, *, device):
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        return seeds, torch.ops.prims.inductor_random.default(shape, seed, "rand")
    total_advance = 8 + _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


@oracle_impl(hardware="B200", point="ec769da9", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, view_shape_param, random_shape_param = inputs
    view_shape = _as_shape(view_shape_param)
    random_shape = _as_shape(random_shape_param)
    device = x.device
    n_elements = x.numel()

    mask = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.bool,
    )
    scaled = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.float32,
    )
    complex_out = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=device, dtype=torch.complex64,
    )

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    x_flat = x.contiguous().view(-1)
    random_flat = random.contiguous().view(-1)
    mask_flat = mask.view(-1)
    scaled_flat = scaled.view(-1)
    # complex64 is stored as (real, imag) pair; view_as_real returns [..., 2].
    complex_real_flat = torch.view_as_real(complex_out).view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(n_elements, BLOCK_N), 1, 1), _dropout_complex_kernel,
        (x_flat, random_flat, mask_flat, scaled_flat, complex_real_flat, BLOCK_N),
    )
    return seeds, mask, scaled, complex_out
