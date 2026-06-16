"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 BERT exact-erf GELU, seed-index-3 Inductor dropout, returned bool keep mask, and final contiguous bf16 view in one storage-linear Triton pointwise kernel with the repro's f32 GELU expression, bf16 GELU rounding before dropout, bf16-rounded random comparison against `0.1`, and bf16 dropout multiply/scale boundaries, whereas Inductor lowers this stochastic multi-output pointwise chain through generic pointwise/RNG scheduling; Inductor cannot do this today because it has no dedicated exact-GELU dropout template that preserves these dtype and RNG boundaries while directly emitting both returned tensors; the fix is NEW_PATTERN: add a stochastic exact-GELU dropout pointwise lowering that covers the mask side output and bf16 materialized view."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 128
HIDDEN = 3072
NUMEL = BATCH * SEQ * HIDDEN
SEED_INDEX = 3
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _bert_gelu_dropout_kernel(
    x_ptr,
    random_or_seed_ptr,
    gt_ptr,
    out_ptr,
    total: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p: tl.constexpr,
    dropout_scale: tl.constexpr,
    use_seeded_rng: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    threshold = tl.full((BLOCK,), dropout_p, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    half = x * 0.5
    erf_arg = x * 0.7071067811865476
    erf_val = tl.math.erf(erf_arg)
    gelu = (half * (erf_val + 1.0)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    dropped = (keep.to(tl.float32) * gelu.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    scaled = (dropped.to(tl.float32) * dropout_scale).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(gt_ptr + offsets, keep, mask=mask)
    tl.store(out_ptr + offsets, scaled, mask=mask)


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


# f9ed8dd6: BERT bf16 exact-GELU + seed-index-3 dropout, [2048,3072].
@oracle_impl(hardware="B200", point="f9ed8dd6", BLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    x, seeds = inputs[:2]
    random_shape = tuple(int(dim) for dim in inputs[3])
    out_shape = tuple(int(dim) for dim in inputs[4])

    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=x.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(NUMEL, BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _bert_gelu_dropout_kernel[grid](
            x,
            seeds,
            gt,
            out,
            total=NUMEL,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=True,
            BLOCK=BLOCK,
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
        _bert_gelu_dropout_kernel[grid](
            x,
            random,
            gt,
            out,
            total=NUMEL,
            seed_index=SEED_INDEX,
            dropout_p=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            use_seeded_rng=False,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return gt, out
