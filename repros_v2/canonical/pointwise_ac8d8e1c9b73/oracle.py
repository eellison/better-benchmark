"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender bf16 SELU plus internally seeded dropout pointwise scope in one storage-linear Triton kernel, including fp32 SELU math with the bf16 activation boundary, `prims.inductor_seeds.default(1)`, f32 random rounded to bf16 before `gt(0.8)`, the returned bool mask, and the returned bf16 scaled activation tensor, whereas Inductor lowers the stochastic producer and activation/dropout pointwise work through generic scheduler regions; Inductor cannot fuse this full returned-output envelope today because generated Inductor RNG and the visible stochastic mask/output boundary split the otherwise simple activation epilogue; the fix is SCHEDULER_FUSION: teach pointwise scheduling to keep generated-seed RNG, bf16 thresholding, and dependent activation stores in one guarded fused pointwise plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra import libdevice

from oracle_harness import oracle_impl


ROWS = 256
COLS = 1024
N_ELEMENTS = ROWS * COLS
SEED_COUNT = 1
SEED_INDEX = 0
THRESHOLD = 0.8
SCALE = 5.000000000000001
_SEEDED_KERNEL_WARMED = False


@triton.autotune(
    configs=[
        triton.Config({"block": 128}, num_warps=4, num_stages=4),
        triton.Config({"block": 256}, num_warps=4, num_stages=4),
        triton.Config({"block": 512}, num_warps=4, num_stages=4),
        triton.Config({"block": 1024}, num_warps=4, num_stages=4),
        triton.Config({"block": 2048}, num_warps=4, num_stages=4),
        triton.Config({"block": 2048}, num_warps=8, num_stages=4),
        triton.Config({"block": 4096}, num_warps=8, num_stages=4),
    ],
    key=["n_elements", "use_seeded_rng"],
)
@triton.jit
def _selu_dropout_kernel(
    x_ptr,
    rng_ptr,
    mask_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    seed_index: tl.constexpr,
    threshold: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
    use_seeded_rng: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    positive = x * 1.0507009873554805
    negative = libdevice.expm1(x * 1.0) * 1.7580993408473766
    selu = tl.where(x > 0.0, positive, negative).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    if use_seeded_rng:
        seed = tl.load(rng_ptr + seed_index)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    else:
        random_bf16 = tl.load(rng_ptr + offsets, mask=valid, other=0.0).to(tl.bfloat16)

    threshold_bf16 = tl.full((block,), threshold, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold_bf16
    dropped = tl.where(keep, selu, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = (dropped.to(tl.float32) * scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(mask_ptr + offsets, keep, mask=valid)
    tl.store(out_ptr + offsets, scaled, mask=valid)


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
    _selu_dropout_kernel[grid](
        x,
        dummy_seed,
        warm_mask,
        warm_out,
        n_elements=N_ELEMENTS,
        seed_index=SEED_INDEX,
        threshold=THRESHOLD,
        scale=SCALE,
        use_seeded_rng=True,
    )
    _SEEDED_KERNEL_WARMED = True


# 0f3e2fa1: (T([256,1024], bf16), S([256,1024]))
@oracle_impl(hardware="B200", point="0f3e2fa1")
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
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _selu_dropout_kernel[grid](
            x,
            seeds,
            mask,
            out,
            n_elements=N_ELEMENTS,
            seed_index=SEED_INDEX,
            threshold=THRESHOLD,
            scale=SCALE,
            use_seeded_rng=True,
        )
    else:
        _warm_seeded_kernel(x, random_shape)
        _, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _selu_dropout_kernel[grid](
            x,
            random,
            mask,
            out,
            n_elements=N_ELEMENTS,
            seed_index=SEED_INDEX,
            threshold=THRESHOLD,
            scale=SCALE,
            use_seeded_rng=False,
        )

    return mask, out
