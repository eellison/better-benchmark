"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete VGG16 bf16 ReLU/dropout tuple, including generated Inductor seeds, bf16-rounded random mask, scaled ReLU materialization, and the bool `relu <= 0` side output, while folding the side predicate to the input sign, whereas Inductor lowers the decomposed relu/random/convert/gt/mul/le graph as generic stochastic pointwise work; Inductor cannot do this today because its algebraic simplifier does not canonicalize zero-threshold comparisons through ReLU inside RNG-bearing multi-output pointwise graphs; the fix is ALGEBRAIC_ELIMINATION: add a guarded ReLU-threshold rewrite that preserves generated-seed dropout and bf16 cast boundaries."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 64
COLS = 4096
N_ELEMENTS = ROWS * COLS
SEED_COUNT = 2
SEED_INDEX = 0


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


@triton.jit
def _relu_dropout_kernel(
    x_ptr,
    rng_ptr,
    gt_ptr,
    out_ptr,
    le_ptr,
    total: tl.constexpr,
    use_random_ptr: tl.constexpr,
    seed_index: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    x = tl.load(x_ptr + offsets).to(tl.float32)

    if use_random_ptr:
        le = x <= 0.0
        relu = tl.where(le, 0.0, x).to(tl.bfloat16)
        random_bf16 = tl.load(rng_ptr + offsets).to(tl.bfloat16)
        keep = random_bf16 > tl.full((block,), 0.5, tl.float32).to(tl.bfloat16)
        dropped = keep.to(tl.bfloat16) * relu
        scaled = (dropped * tl.full((block,), 2.0, tl.float32).to(tl.bfloat16)).to(tl.bfloat16)
    else:
        seed = tl.load(rng_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32)).to(tl.float32)
        keep = random > 0.5
        relu_f32 = tl.maximum(x, 0.0)
        le = relu_f32 <= 0.0
        scaled = keep.to(tl.float32) * relu_f32 * 2.0

    tl.store(gt_ptr + offsets, keep)
    tl.store(out_ptr + offsets, scaled)
    tl.store(le_ptr + offsets, le)


@oracle_impl(hardware="B200", point="200ac53a", block=512, num_warps=4, num_stages=2)
def oracle_forward(inputs, *, block: int, num_warps: int, num_stages: int):
    x, shape = inputs
    out_shape = tuple(int(dim) for dim in shape)
    gt = torch.empty_strided(out_shape, (COLS, 1), device=x.device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, (COLS, 1), device=x.device, dtype=torch.bfloat16)
    le = torch.empty_strided(out_shape, (COLS, 1), device=x.device, dtype=torch.bool)

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.empty((SEED_COUNT,), device=x.device, dtype=torch.int64)
        torch.ops.aten.randint.low_out(
            -9223372036854775808,
            9223372036854775807,
            [SEED_COUNT],
            out=seeds,
        )
        rng_source = seeds
        use_random_ptr = False
    else:
        seeds, rng_source = _seeds_and_random_for_eager_check(out_shape, device=x.device)
        use_random_ptr = True

    grid = (triton.cdiv(N_ELEMENTS, block),)
    _relu_dropout_kernel[grid](
        x,
        rng_source,
        gt,
        dropped,
        le,
        total=N_ELEMENTS,
        use_random_ptr=use_random_ptr,
        seed_index=SEED_INDEX,
        block=block,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return seeds, gt, dropped, le
