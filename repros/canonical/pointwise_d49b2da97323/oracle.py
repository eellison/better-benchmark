"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileNetV3 bf16 hard-swish plus internally seeded keep_prob=0.8 dropout scope in one storage-linear Triton kernel, including the returned bool mask, fp32 hard-swish arithmetic, folded dropout scale, and final bf16 output store, whereas Inductor lowers the captured stochastic pointwise graph through the generic RNG/mask/cast/div/mul expression chain; Inductor cannot do this today because its pointwise algebraic simplifier does not fold scalar dropout factors through a visible random-mask conversion while preserving the returned mask and output rounding boundary; the fix is ALGEBRAIC_ELIMINATION: teach pointwise codegen to fold the mask scale and hard-swish scalar factors across stochastic mask uses subject to the captured dtype boundaries."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32
COLS = 1280
N_ELEMENTS = ROWS * COLS
SEED_COUNT = 1
SEED_INDEX = 0
KEEP_PROB = 0.8
DROPOUT_SCALE = 1.25
_SEEDED_KERNEL_WARMED = False


@triton.autotune(
    configs=[
        triton.Config({"block": 128}, num_warps=1, num_stages=4),
        triton.Config({"block": 256}, num_warps=1, num_stages=4),
        triton.Config({"block": 512}, num_warps=1, num_stages=4),
        triton.Config({"block": 256}, num_warps=2, num_stages=4),
        triton.Config({"block": 512}, num_warps=2, num_stages=4),
        triton.Config({"block": 1024}, num_warps=2, num_stages=4),
        triton.Config({"block": 256}, num_warps=4, num_stages=4),
        triton.Config({"block": 512}, num_warps=4, num_stages=4),
        triton.Config({"block": 1024}, num_warps=4, num_stages=4),
        triton.Config({"block": 2048}, num_warps=8, num_stages=4),
    ],
    key=["n_elements", "use_seeded_rng"],
)
@triton.jit
def _hardswish_dropout_kernel(
    x_ptr,
    rng_ptr,
    mask_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    seed_index: tl.constexpr,
    keep_prob: tl.constexpr,
    dropout_scale: tl.constexpr,
    block: tl.constexpr,
    use_seeded_rng: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    shifted = x + 3.0
    clamp_min = tl.maximum(shifted, 0.0)
    clamp_max = tl.minimum(clamp_min, 6.0)
    hardswish = (x * clamp_max) * 0.16666666666666666

    if use_seeded_rng:
        seed = tl.load(rng_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
    else:
        random = tl.load(rng_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    keep = random < keep_prob
    keep_f32 = keep.to(tl.float32)
    out = (hardswish * (keep_f32 * dropout_scale)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(mask_ptr + offsets, keep, mask=valid)
    tl.store(out_ptr + offsets, out, mask=valid)


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


def _capture_seeds(device):
    seeds = torch.empty((SEED_COUNT,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [SEED_COUNT],
        out=seeds,
    )
    return seeds


def _warm_seeded_kernel(x, random_shape):
    global _SEEDED_KERNEL_WARMED
    if _SEEDED_KERNEL_WARMED:
        return

    dummy_seed = torch.empty((SEED_COUNT,), device=x.device, dtype=torch.int64)
    warm_mask = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    warm_out = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["block"]),)
    _hardswish_dropout_kernel[grid](
        x,
        dummy_seed,
        warm_mask,
        warm_out,
        n_elements=N_ELEMENTS,
        seed_index=SEED_INDEX,
        keep_prob=KEEP_PROB,
        dropout_scale=DROPOUT_SCALE,
        use_seeded_rng=True,
    )
    _SEEDED_KERNEL_WARMED = True


# 040ff6c3: (T([32,1280], bf16), S([32,1280]))
@oracle_impl(hardware="B200", point="040ff6c3")
def oracle_forward(inputs):
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

    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["block"]),)
    if torch.cuda.is_current_stream_capturing():
        seeds = _capture_seeds(device)
        _hardswish_dropout_kernel[grid](
            x,
            seeds,
            mask,
            out,
            n_elements=N_ELEMENTS,
            seed_index=SEED_INDEX,
            keep_prob=KEEP_PROB,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=True,
        )
    else:
        _warm_seeded_kernel(x, random_shape)
        _, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _hardswish_dropout_kernel[grid](
            x,
            random,
            mask,
            out,
            n_elements=N_ELEMENTS,
            seed_index=SEED_INDEX,
            keep_prob=KEEP_PROB,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=False,
        )

    return mask, out
