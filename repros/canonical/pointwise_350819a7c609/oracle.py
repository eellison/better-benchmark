"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 BERT dual-dropout residual scope in one Triton pointwise kernel, including the `[2048,768] -> [16,128,768]` metadata view, seed-index-59 bf16-rounded RNG compare, returned first bool mask, bf16 mask multiply and scale rounding before the fp32 residual add, seed-index-60 fp32 RNG compare, returned second bool mask, fp32 dropout scaling, selected token bf16 side output, and flattened bf16 view output; Inductor lowers the two stochastic producers and sibling pointwise/view/select consumers through generic pointwise scheduling instead of one full-output plan; the fix is SCHEDULER_FUSION: teach the pointwise scheduler to inline multiple seeded `prims.inductor_random` streams with the exact dtype boundaries and emit all side outputs from a single fused kernel."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
HIDDEN = 768
TOTAL = BATCH * SEQ * HIDDEN
SEED_INDEX_0 = 59
SEED_INDEX_1 = 60
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _dual_dropout_kernel(
    x_ptr,
    seeds_or_random0_ptr,
    random1_ptr,
    residual_ptr,
    mask0_ptr,
    mask1_ptr,
    selected_ptr,
    out_ptr,
    total: tl.constexpr,
    seed_index0: tl.constexpr,
    seed_index1: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_seeded_rng: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < total

    if use_seeded_rng:
        seed0 = tl.load(seeds_or_random0_ptr + seed_index0)
        seed1 = tl.load(seeds_or_random0_ptr + seed_index1)
        random0 = tl.rand(seed0, offsets.to(tl.uint32))
        random1 = tl.rand(seed1, offsets.to(tl.uint32))
    else:
        random0 = tl.load(
            seeds_or_random0_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        )
        random1 = tl.load(
            random1_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        )

    threshold_bf16 = tl.full((BLOCK_N,), dropout_p, tl.float32).to(tl.bfloat16)
    keep0 = random0.to(tl.bfloat16) > threshold_bf16
    keep1 = random1 > dropout_p

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    dropped0 = _f32_mul(keep0.to(tl.float32), x).to(tl.bfloat16)
    scaled0 = _f32_mul(dropped0.to(tl.float32), dropout_scale).to(tl.bfloat16)
    added = _f32_add(residual, scaled0.to(tl.float32))
    dropped1 = _f32_mul(keep1.to(tl.float32), added)
    final = _f32_mul(dropped1, dropout_scale)
    final_bf16 = final.to(tl.bfloat16)

    tl.store(mask0_ptr + offsets, keep0, mask=valid)
    tl.store(mask1_ptr + offsets, keep1, mask=valid)
    tl.store(out_ptr + offsets, final_bf16, mask=valid)

    hidden = offsets % hidden_size
    row = offsets // hidden_size
    seq = row % seq_len
    batch = row // seq_len
    selected_offsets = batch * hidden_size + hidden
    tl.store(selected_ptr + selected_offsets, final_bf16, mask=valid & (seq == 0))


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


def _launch(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
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

    random_shape = tuple(int(dim) for dim in random_shape0)
    flat_out_shape = tuple(int(dim) for dim in out_shape)
    mask0 = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    mask1 = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    selected = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        flat_out_shape,
        _contiguous_stride(flat_out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(TOTAL, BLOCK_N),)
    if torch.cuda.is_current_stream_capturing():
        _dual_dropout_kernel[grid](
            x,
            seeds,
            seeds,
            residual,
            mask0,
            mask1,
            selected,
            out,
            total=TOTAL,
            seed_index0=SEED_INDEX_0,
            seed_index1=SEED_INDEX_1,
            seq_len=SEQ,
            hidden_size=HIDDEN,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=True,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
        seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
        random0, random1 = _inductor_random_pair_for_eager_check(
            random_shape,
            seed0,
            seed1,
            device=x.device,
        )
        _dual_dropout_kernel[grid](
            x,
            random0,
            random1,
            residual,
            mask0,
            mask1,
            selected,
            out,
            total=TOTAL,
            seed_index0=SEED_INDEX_0,
            seed_index1=SEED_INDEX_1,
            seq_len=SEQ,
            hidden_size=HIDDEN,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=False,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return mask0, mask1, selected, out


# 4a069931: (T([2048,768], bf16), T([61], i64), T([16,128,768], f32), ...)
@oracle_impl(hardware="B200", point="4a069931", BLOCK_N=256, num_warps=4, num_stages=3)
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
