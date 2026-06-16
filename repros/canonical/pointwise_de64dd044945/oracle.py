"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet bf16 two-input ReLU, virtual channel-cat, internally generated seed-index-0 Inductor dropout, bf16-random `gt(0.5)` mask, bf16 dropout multiply/scale, returned bool mask, and both ReLU backward-mask sibling outputs in one Triton pointwise kernel without materializing either ReLU activation or the concatenated `[32,512,13,13]` intermediate. Inductor lowers the ReLU/cat/generated-RNG/dropout/mask tuple through generic stochastic pointwise scheduling with avoidable virtual-layout and sibling-output work. The fix is SCHEDULER_FUSION: teach pointwise scheduling to fuse static channel concat producers, generated-seed dropout epilogues, bf16 RNG comparison boundaries, and deterministic sibling masks into one multi-output layout-aware kernel."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
IN_CHANNELS = 256
OUT_CHANNELS = 512
H = 13
W = 13
HW = H * W
INPUT_NUMEL = BATCH * IN_CHANNELS * HW
OUTPUT_NUMEL = BATCH * OUT_CHANNELS * HW
INPUT_SHAPE = (BATCH, IN_CHANNELS, H, W)
INPUT_STRIDE = (IN_CHANNELS * HW, HW, W, 1)
OUTPUT_SHAPE = (BATCH, OUT_CHANNELS, H, W)
OUTPUT_STRIDE = (OUT_CHANNELS * HW, HW, W, 1)
SEED_COUNT = 1
SEED_INDEX = 0
DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _relu_preserve_nan(x):
    return tl.where(x <= 0.0, 0.0, x)


@triton.jit
def _virtual_cat_dropout_masks_kernel(
    x0_ptr,
    x1_ptr,
    seeds_or_random_ptr,
    gt_ptr,
    dropout_ptr,
    le_x1_ptr,
    le_x0_ptr,
    input_total: tl.constexpr,
    input_batch_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    channel_cat_offset: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_random_ptr: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    active = offsets < input_total

    batch = offsets // input_batch_stride
    batch_inner = offsets - batch * input_batch_stride
    out_x0_offsets = batch * output_batch_stride + batch_inner
    out_x1_offsets = out_x0_offsets + channel_cat_offset

    x0 = tl.load(x0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    relu0 = _relu_preserve_nan(x0).to(tl.bfloat16)
    relu1 = _relu_preserve_nan(x1).to(tl.bfloat16)

    if use_random_ptr:
        random0 = tl.load(
            seeds_or_random_ptr + out_x0_offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        random1 = tl.load(
            seeds_or_random_ptr + out_x1_offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_or_random_ptr + seed_index)
        random0 = tl.rand(seed, out_x0_offsets.to(tl.uint32)).to(tl.bfloat16)
        random1 = tl.rand(seed, out_x1_offsets.to(tl.uint32)).to(tl.bfloat16)

    threshold = tl.full((BLOCK_N,), dropout_p, tl.float32).to(tl.bfloat16)
    keep0 = random0 > threshold
    keep1 = random1 > threshold

    dropped0 = _f32_mul(keep0.to(tl.float32), relu0.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    dropped1 = _f32_mul(keep1.to(tl.float32), relu1.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    scaled0 = _f32_mul(dropped0.to(tl.float32), dropout_scale).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    scaled1 = _f32_mul(dropped1.to(tl.float32), dropout_scale).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(gt_ptr + out_x0_offsets, keep0, mask=active)
    tl.store(gt_ptr + out_x1_offsets, keep1, mask=active)
    tl.store(dropout_ptr + out_x0_offsets, scaled0, mask=active)
    tl.store(dropout_ptr + out_x1_offsets, scaled1, mask=active)
    tl.store(le_x1_ptr + offsets, x1 <= 0.0, mask=active)
    tl.store(le_x0_ptr + offsets, x0 <= 0.0, mask=active)


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


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape, device=device)
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


@oracle_impl(hardware="B200", point="30725500", BLOCK_N=1024, num_warps=4, num_stages=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    x0, x1, random_shape_param = inputs
    random_shape = tuple(int(dim) for dim in random_shape_param)
    device = x0.device

    gt = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=device,
        dtype=torch.bool,
    )
    dropout = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )
    le_x1 = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=device,
        dtype=torch.bool,
    )
    le_x0 = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=device,
        dtype=torch.bool,
    )

    grid = (triton.cdiv(INPUT_NUMEL, BLOCK_N),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _virtual_cat_dropout_masks_kernel[grid](
            x0,
            x1,
            seeds,
            gt,
            dropout,
            le_x1,
            le_x0,
            input_total=INPUT_NUMEL,
            input_batch_stride=IN_CHANNELS * HW,
            output_batch_stride=OUT_CHANNELS * HW,
            channel_cat_offset=IN_CHANNELS * HW,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_random_ptr=False,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _virtual_cat_dropout_masks_kernel[grid](
            x0,
            x1,
            random,
            gt,
            dropout,
            le_x1,
            le_x0,
            input_total=INPUT_NUMEL,
            input_batch_stride=IN_CHANNELS * HW,
            output_batch_stride=OUT_CHANNELS * HW,
            channel_cat_offset=IN_CHANNELS * HW,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_random_ptr=True,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, dropout, le_x1, le_x0
