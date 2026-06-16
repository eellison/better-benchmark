"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XLNet training token-embedding plus positional-dropout fanout scope with two Triton pointwise kernels, including the returned transposed token-id clone, internally generated `prims.inductor_seeds.default(99)`, seed-index-0 f32 embedding dropout, returned mask and scaled f32 tensor, bf16 view/permute alias pair, seed-index-1 bf16-threshold positional dropout over the sinusoidal table, and the second bf16 view/permute alias pair, whereas Inductor lowers the embedding gather, generated RNG producers, sinusoidal table, dropout chains, and visible layout aliases through separate generic pointwise/layout schedules; Inductor cannot do this today because pointwise scheduling has no guarded XLNet template that keeps generated seeds, indexed embedding loads, structured sin/cos values, stochastic masks, and alias-preserving output stores in one full returned-output plan; the fix is NEW_PATTERN: add an XLNet embedding-plus-positional-dropout lowering that fuses the gathered token path and shape-derived positional path while preserving RNG streams, dtype rounding boundaries, and returned alias metadata."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 512
POS_SEQ = 1024
HIDDEN = 1024
HALF_HIDDEN = 512
TOKEN_ROWS = SEQ * BATCH
POS_ROWS = POS_SEQ * BATCH
TOKEN_NUMEL = TOKEN_ROWS * HIDDEN
POS_NUMEL = POS_ROWS * HIDDEN
SEED_COUNT = 99
SEED_INDEX_TOKEN = 0
SEED_INDEX_POS = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
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
def _token_embedding_dropout_kernel(
    ids_ptr,
    table_ptr,
    rng_ptr,
    clone_ptr,
    mask_ptr,
    scaled_ptr,
    bf16_ptr,
    N: tl.constexpr,
    SEQ_C: tl.constexpr,
    BATCH_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    SEED_INDEX_C: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < N

    hidden = offsets % HIDDEN_C
    row = offsets // HIDDEN_C
    batch = row % BATCH_C
    seq = row // BATCH_C
    token_id = tl.load(ids_ptr + batch * SEQ_C + seq, mask=active, other=0)
    embedding = tl.load(
        table_ptr + token_id * HIDDEN_C + hidden,
        mask=active,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_RANDOM_PTR:
        random = tl.load(
            rng_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed = tl.load(rng_ptr + SEED_INDEX_C)
        random = tl.rand(seed, offsets.to(tl.uint32))

    keep = random > DROPOUT_P_C
    scaled = _f32_mul(_f32_mul(keep.to(tl.float32), embedding), DROPOUT_SCALE_C)

    tl.store(clone_ptr + row, token_id, mask=active & (hidden == 0))
    tl.store(mask_ptr + offsets, keep, mask=active)
    tl.store(scaled_ptr + offsets, scaled, mask=active)
    tl.store(
        bf16_ptr + offsets,
        scaled.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )


@triton.jit
def _positional_dropout_kernel(
    rng_ptr,
    out_ptr,
    N: tl.constexpr,
    BATCH_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
    HALF_HIDDEN_C: tl.constexpr,
    SEED_INDEX_C: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < N

    hidden = offsets % HIDDEN_C
    row = offsets // HIDDEN_C
    pos_index = row // BATCH_C
    freq_index = tl.where(hidden < HALF_HIDDEN_C, hidden, hidden - HALF_HIDDEN_C)

    pos = (512 - pos_index).to(tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    even_dim = _f32_mul(freq_index.to(tl.float32), 2.0)
    exponent = _f32_div(even_dim, 1024.0)
    denom = libdevice.pow(tl.full((BLOCK,), 10000.0, tl.float32), exponent)
    inv_freq = _f32_div(1.0, denom).to(tl.bfloat16, fp_downcast_rounding="rtne")
    phase = _f32_mul(pos.to(tl.float32), inv_freq.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    trig = tl.where(
        hidden < HALF_HIDDEN_C,
        libdevice.sin(phase.to(tl.float32)),
        libdevice.cos(phase.to(tl.float32)),
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")

    if USE_RANDOM_PTR:
        random_bf16 = tl.load(
            rng_ptr + offsets,
            mask=active,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(rng_ptr + SEED_INDEX_C)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)

    threshold = tl.full((BLOCK,), DROPOUT_P_C, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    dropped = _f32_mul(keep.to(tl.float32), trig.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_C).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + offsets, scaled, mask=active)


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


def _seeds_and_randoms_for_eager_check(shape0, shape1, *, device):
    total_advance = 8 + _random_advance(shape0) + _random_advance(shape1)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_TOKEN)
    random0 = torch.ops.prims.inductor_random.default(shape0, seed0, "rand")
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_POS)
    random1 = torch.ops.prims.inductor_random.default(shape1, seed1, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random0, random1


def _allocate_outputs(device):
    clone = torch.empty_strided(
        (SEQ, BATCH),
        (BATCH, 1),
        device=device,
        dtype=torch.int64,
    )
    mask = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.bool,
    )
    scaled = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    token_bf16 = torch.empty_strided(
        (TOKEN_ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    pos_bf16 = torch.empty_strided(
        (POS_ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    return clone, mask, scaled, token_bf16, pos_bf16


def _launch_kernels(
    arg0_1,
    arg1_1,
    rng0,
    rng1,
    clone,
    mask,
    scaled,
    token_bf16,
    pos_bf16,
    *,
    use_random_ptr,
    TOKEN_BLOCK,
    POS_BLOCK,
    token_warps,
    pos_warps,
    num_stages,
):
    _token_embedding_dropout_kernel[(triton.cdiv(TOKEN_NUMEL, TOKEN_BLOCK),)](
        arg0_1,
        arg1_1,
        rng0,
        clone,
        mask,
        scaled,
        token_bf16,
        N=TOKEN_NUMEL,
        SEQ_C=SEQ,
        BATCH_C=BATCH,
        HIDDEN_C=HIDDEN,
        SEED_INDEX_C=SEED_INDEX_TOKEN,
        DROPOUT_P_C=DROPOUT_P,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK=TOKEN_BLOCK,
        num_warps=token_warps,
        num_stages=num_stages,
    )
    _positional_dropout_kernel[(triton.cdiv(POS_NUMEL, POS_BLOCK),)](
        rng1,
        pos_bf16,
        N=POS_NUMEL,
        BATCH_C=BATCH,
        HIDDEN_C=HIDDEN,
        HALF_HIDDEN_C=HALF_HIDDEN,
        SEED_INDEX_C=SEED_INDEX_POS,
        DROPOUT_P_C=DROPOUT_P,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK=POS_BLOCK,
        num_warps=pos_warps,
        num_stages=num_stages,
    )


# 4ae3e770: XLNetLMHeadModel train embedding + dual dropout positional fanout.
@oracle_impl(
    hardware="B200",
    point="4ae3e770",
    TOKEN_BLOCK=1024,
    POS_BLOCK=512,
    token_warps=4,
    pos_warps=4,
    num_stages=4,
)
def oracle_forward(
    inputs,
    *,
    TOKEN_BLOCK: int,
    POS_BLOCK: int,
    token_warps: int,
    pos_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, _shape0, _shape1, _shape2, _shape3, _shape4 = inputs
    del _shape0, _shape1, _shape2, _shape3, _shape4

    clone, mask, scaled, token_bf16, pos_bf16 = _allocate_outputs(arg1_1.device)

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, arg1_1.device)
        _launch_kernels(
            arg0_1,
            arg1_1,
            seeds,
            seeds,
            clone,
            mask,
            scaled,
            token_bf16,
            pos_bf16,
            use_random_ptr=False,
            TOKEN_BLOCK=TOKEN_BLOCK,
            POS_BLOCK=POS_BLOCK,
            token_warps=token_warps,
            pos_warps=pos_warps,
            num_stages=num_stages,
        )
        return (
            clone,
            seeds,
            mask,
            scaled,
            token_bf16,
            pos_bf16,
            pos_bf16.t(),
            token_bf16.t(),
        )

    seeds, random0, random1 = _seeds_and_randoms_for_eager_check(
        (SEQ, BATCH, HIDDEN),
        (POS_SEQ, BATCH, HIDDEN),
        device=arg1_1.device,
    )
    _launch_kernels(
        arg0_1,
        arg1_1,
        seeds,
        seeds,
        clone,
        mask,
        scaled,
        token_bf16,
        pos_bf16,
        use_random_ptr=False,
        TOKEN_BLOCK=TOKEN_BLOCK,
        POS_BLOCK=POS_BLOCK,
        token_warps=token_warps,
        pos_warps=pos_warps,
        num_stages=num_stages,
    )
    _launch_kernels(
        arg0_1,
        arg1_1,
        random0,
        random1,
        clone,
        mask,
        scaled,
        token_bf16,
        pos_bf16,
        use_random_ptr=True,
        TOKEN_BLOCK=TOKEN_BLOCK,
        POS_BLOCK=POS_BLOCK,
        token_warps=token_warps,
        pos_warps=pos_warps,
        num_stages=num_stages,
    )
    return (
        clone,
        seeds,
        mask,
        scaled,
        token_bf16,
        pos_bf16,
        pos_bf16.t(),
        token_bf16.t(),
    )
