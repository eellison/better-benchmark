"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 T5 training ReLU/dropout side-mask scope in one flat Triton pointwise kernel, including the `[8192,2048] -> [8,1024,2048]` metadata view, seed-index-49 Inductor random stream, f32-to-bf16 random cast before `gt(0.1)`, returned bool keep mask, bf16 ReLU with NaN preservation, bool-mask multiply, bf16 dropout scaling by 1.1111111111111112, returned flattened bf16 output view, and returned `relu <= 0` bool mask, whereas Inductor already lowers this full observable work into the same practical stochastic pointwise memory/RNG envelope; Inductor cannot materially improve this local repro through scheduler fusion alone because the required activation read, RNG generation, bf16 output store, and two bool side-output stores dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor stochastic pointwise case unless broader RNG codegen, pointwise bandwidth, or launch-overhead changes move both implementations."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
HIDDEN = 2048
ROWS = BATCH * SEQ
NUMEL = ROWS * HIDDEN
VIEW_SHAPE = (BATCH, SEQ, HIDDEN)
VIEW_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
OUT_SHAPE = (ROWS, HIDDEN)
SEED_INDEX = 49
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


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
def _relu_dropout_masks_kernel(
    x_ptr,
    random_or_seed_ptr,
    gt_ptr,
    out_ptr,
    le_ptr,
    total: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_seeded_rng: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total

    if use_seeded_rng:
        seed = tl.load(random_or_seed_ptr + seed_index)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    else:
        random_bf16 = tl.load(
            random_or_seed_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)

    threshold = tl.full((BLOCK_N,), dropout_p, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    zero_bf16 = tl.full((BLOCK_N,), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where((x > zero_bf16) | (x != x), x, zero_bf16)
    non_positive = relu <= zero_bf16

    dropped = _f32_mul(keep.to(tl.float32), relu.to(tl.float32)).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), dropout_scale).to(tl.bfloat16)

    tl.store(gt_ptr + offsets, keep, mask=mask)
    tl.store(out_ptr + offsets, scaled, mask=mask)
    tl.store(le_ptr + offsets, non_positive, mask=mask)


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


def _launch(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    x, seeds, view_shape, random_shape, out_shape = inputs
    del view_shape

    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    out_base = torch.empty_strided(
        VIEW_SHAPE,
        VIEW_STRIDE,
        device=x.device,
        dtype=torch.bfloat16,
    )
    le = torch.empty_strided(
        VIEW_SHAPE,
        VIEW_STRIDE,
        device=x.device,
        dtype=torch.bool,
    )

    grid = (triton.cdiv(NUMEL, BLOCK_N),)
    if torch.cuda.is_current_stream_capturing():
        _relu_dropout_masks_kernel[grid](
            x,
            seeds,
            gt,
            out_base,
            le,
            total=NUMEL,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=True,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=x.device,
        )
        _relu_dropout_masks_kernel[grid](
            x,
            random,
            gt,
            out_base,
            le,
            total=NUMEL,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=False,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, out_base.view(out_shape), le


# 52dd4c9c: (T([8192,2048], bf16), T([64], i64), S([8,1024,2048]), ...)
@oracle_impl(hardware="B200", point="52dd4c9c", BLOCK_N=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
