"""cuTile port of pointwise_d49b2da97323: MobileNetV3 hard-swish + dropout(0.8).

The eager Repro internally generates seeds via inductor_seeds+inductor_random.
For the eager (non-graph-capture) numerics gate, we mirror the Triton oracle's
rewind-and-restore of CUDA RNG state so we regenerate the same random tensor.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32
COLS = 1280
N_ELEMENTS = ROWS * COLS
SEED_COUNT = 1
SEED_INDEX = 0
KEEP_PROB = 0.8
DROPOUT_SCALE = 1.25


@ct.kernel
def _hardswish_dropout_kernel(
    x_ptr,           # bf16 [N_ELEMENTS]
    random_ptr,      # f32  [N_ELEMENTS]
    mask_ptr,        # b8   [N_ELEMENTS]
    out_ptr,         # bf16 [N_ELEMENTS]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(
        x_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = ct.astype(x_bf, ct.float32)
    shifted = x + 3.0
    # clamp(x+3, 0, 6)
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    six = ct.full(shape=(BLOCK,), fill_value=6.0, dtype=ct.float32)
    clamp_min = ct.where(shifted > zero, shifted, zero)
    clamp_max = ct.where(clamp_min < six, clamp_min, six)
    hardswish = (x * clamp_max) * 0.16666666666666666

    random = ct.load(
        random_ptr, index=(pid,), shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep = random < KEEP_PROB
    keep_f = ct.astype(keep, ct.float32)
    out_f = hardswish * (keep_f * DROPOUT_SCALE)
    out_bf = ct.astype(out_f, ct.bfloat16)

    ct.store(mask_ptr, index=(pid,), tile=keep)
    ct.store(out_ptr, index=(pid,), tile=out_bf)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return max(8, (numel + 131071) // 131072)


def _seeds_and_random_for_eager_check(shape, *, device):
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


@oracle_impl(hardware="B200", point="040ff6c3", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    x, random_shape_param = inputs
    random_shape = _shape_tuple(random_shape_param)
    device = x.device

    mask = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    _, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    x_flat = x.view(N_ELEMENTS)
    random_flat = random.reshape(N_ELEMENTS).contiguous()
    mask_flat = mask.view(N_ELEMENTS)
    out_flat = out.view(N_ELEMENTS)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(N_ELEMENTS, BLOCK), 1, 1)
    ct.launch(
        stream, grid, _hardswish_dropout_kernel,
        (x_flat, random_flat, mask_flat, out_flat, BLOCK),
    )
    return mask, out
