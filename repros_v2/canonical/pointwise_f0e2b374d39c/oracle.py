"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete AlexNet bf16 ReLU, 3x3 stride-2 low-memory maxpool values and int8 offsets, shape-identical adaptive_avg_pool2d/view, internally generated seed-index-0 dropout with f32-random-to-bf16 thresholding at `> 0.5`, returned seed and bool mask tensors, bf16 dropout scaling by 2.0, and the full input-shaped `relu <= 0` side mask using one pool/dropout stencil kernel plus one flat mask kernel, whereas Inductor already lowers the same observable scope into the practical stencil/dropout work and a simple sibling mask store with no large removable intermediate after the identity pool/view; Inductor cannot materially improve this local repro because the remaining cost is dominated by required overlapping maxpool reads, RNG/mask/dropout stores, int8 offset stores, and the full bool mask rather than a missed algebraic elimination or layout fusion; the fix is BANDWIDTH_BOUND: record this as an at-floor stochastic stencil/pointwise case unless broader maxpool/RNG scheduling changes move both paths."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 256
H_IN = 13
W_IN = 13
H_OUT = 6
W_OUT = 6
INPUT_NUMEL = BATCH * CHANNELS * H_IN * W_IN
POOL_NUMEL = BATCH * CHANNELS * H_OUT * W_OUT
SEED_COUNT = 2
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
def _pool_dropout_kernel(
    input_ptr,
    seeds_or_random_ptr,
    values_ptr,
    offsets_ptr,
    gt_ptr,
    dropout_ptr,
    total_pool: tl.constexpr,
    channels: tl.constexpr,
    h_in: tl.constexpr,
    w_in: tl.constexpr,
    h_out: tl.constexpr,
    w_out: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_random_ptr: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    active = out_offsets < total_pool

    ow = out_offsets % w_out
    tmp = out_offsets // w_out
    oh = tmp % h_out
    tmp = tmp // h_out
    channel = tmp % channels
    batch = tmp // channels
    input_base = (batch * channels + channel) * (h_in * w_in)

    best = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    best_offset = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for kh in tl.static_range(0, 3):
        ih = oh * 2 + kh
        for kw in tl.static_range(0, 3):
            iw = ow * 2 + kw
            raw = tl.load(
                input_ptr + input_base + ih * w_in + iw,
                mask=active,
                other=-float("inf"),
            ).to(tl.float32)
            relu = _relu_preserve_nan(raw)
            take = active & ((relu > best) | (relu != relu))
            best = tl.where(take, relu, best)
            best_offset = tl.where(take, kh * 3 + kw, best_offset)

    pooled = best.to(tl.bfloat16, fp_downcast_rounding="rtne")
    if use_random_ptr:
        random_bf16 = tl.load(
            seeds_or_random_ptr + out_offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_or_random_ptr + seed_index)
        random_bf16 = tl.rand(seed, out_offsets.to(tl.uint32)).to(tl.bfloat16)

    threshold = tl.full((BLOCK_N,), dropout_p, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    dropped = _f32_mul(keep.to(tl.float32), pooled.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    scaled = _f32_mul(dropped.to(tl.float32), dropout_scale).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(values_ptr + out_offsets, pooled, mask=active)
    tl.store(offsets_ptr + out_offsets, best_offset.to(tl.int8), mask=active)
    tl.store(gt_ptr + out_offsets, keep, mask=active)
    tl.store(dropout_ptr + out_offsets, scaled, mask=active)


@triton.jit
def _relu_le_mask_kernel(
    input_ptr,
    le_ptr,
    total_input: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    active = offsets < total_input
    raw = tl.load(input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    tl.store(le_ptr + offsets, raw <= 0.0, mask=active)


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


@oracle_impl(
    hardware="B200",
    point="a8ee30c6",
    POOL_BLOCK=256,
    MASK_BLOCK=1024,
    pool_warps=8,
    mask_warps=4,
)
def oracle_forward(
    inputs,
    *,
    POOL_BLOCK: int,
    MASK_BLOCK: int,
    pool_warps: int,
    mask_warps: int,
):
    x, _kernel_size, _stride, flat_shape_param, random_shape_param = inputs
    flat_shape = tuple(int(dim) for dim in flat_shape_param)
    random_shape = tuple(int(dim) for dim in random_shape_param)
    device = x.device

    values = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    offsets = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * H_OUT * W_OUT, H_OUT * W_OUT, W_OUT, 1),
        device=device,
        dtype=torch.int8,
    )
    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    dropout = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=device,
        dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        (BATCH, CHANNELS, H_IN, W_IN),
        (CHANNELS * H_IN * W_IN, H_IN * W_IN, W_IN, 1),
        device=device,
        dtype=torch.bool,
    )

    pool_grid = (triton.cdiv(POOL_NUMEL, POOL_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _pool_dropout_kernel[pool_grid](
            x,
            seeds,
            values,
            offsets,
            gt,
            dropout,
            total_pool=POOL_NUMEL,
            channels=CHANNELS,
            h_in=H_IN,
            w_in=W_IN,
            h_out=H_OUT,
            w_out=W_OUT,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_random_ptr=False,
            BLOCK_N=POOL_BLOCK,
            num_warps=pool_warps,
            num_stages=4,
        )
    else:
        seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _pool_dropout_kernel[pool_grid](
            x,
            random,
            values,
            offsets,
            gt,
            dropout,
            total_pool=POOL_NUMEL,
            channels=CHANNELS,
            h_in=H_IN,
            w_in=W_IN,
            h_out=H_OUT,
            w_out=W_OUT,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_random_ptr=True,
            BLOCK_N=POOL_BLOCK,
            num_warps=pool_warps,
            num_stages=4,
        )

    _relu_le_mask_kernel[(triton.cdiv(INPUT_NUMEL, MASK_BLOCK),)](
        x,
        le,
        total_input=INPUT_NUMEL,
        BLOCK_N=MASK_BLOCK,
        num_warps=mask_warps,
        num_stages=4,
    )
    return values, offsets, seeds, gt, dropout, le
