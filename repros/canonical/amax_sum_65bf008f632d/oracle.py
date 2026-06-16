"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DeBERTa masked attention softmax/dropout scope in one Triton row kernel, including the `[192,512,512]` to `[8,24,512,512]` view, broadcast bool mask with bf16 scalar fill, returned bf16 masked-score tensor, fp32 stable last-dimension amax/libdevice.exp/sum/div side outputs, Inductor seed-index-34 dropout mask, f32 dropout scaling before the final bf16 cast, returned contiguous `[192,512,512]` view, and returned permute alias, whereas Inductor lowers the decomposed view/where/cast/amax/sub/exp/sum/div/RNG/dropout/view/cast/permute graph through separate generic mask, reduction, stochastic producer, and layout fragments; Inductor cannot do this today because its scheduler does not keep the seeded dropout producer, observable reduction side outputs, masked-score side output, and alias-producing layout epilogue resident inside one row-softmax plan while preserving bf16 and f32 dtype boundaries; the fix is SCHEDULER_FUSION: teach the attention-softmax scheduler to fuse scalar-fill masking, normalization, Inductor-seeded dropout, side-output stores, final cast, and permute-view emission."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 34


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
def _masked_softmax_dropout_random_kernel(
    x_ptr,
    mask_ptr,
    fill_ptr,
    random_ptr,
    masked_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    dropped_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    query = rows - flat_bh * Q_LEN
    offsets = rows[:, None] * K_LEN + cols[None, :]
    mask_offsets = (
        batch[:, None] * (Q_LEN * K_LEN)
        + query[:, None] * K_LEN
        + cols[None, :]
    )

    x = tl.load(x_ptr + offsets, mask=active, other=0.0)
    mask = tl.load(mask_ptr + mask_offsets, mask=active, other=0).to(tl.int1)
    fill = tl.load(fill_ptr)
    masked = tl.where(mask, fill, x)
    tl.store(masked_ptr + offsets, masked, mask=active)

    scores = tl.where(active, masked.to(tl.float32), -float("inf"))
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    random = tl.load(random_ptr + offsets, mask=active, other=0.0)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32)
    keep = random > dropout_p
    tl.store(keep_ptr + offsets, keep, mask=active)

    dropped = tl.where(keep, probs, 0.0)
    scaled = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled, mask=active)


@triton.jit
def _masked_softmax_dropout_seeded_kernel(
    x_ptr,
    mask_ptr,
    fill_ptr,
    seeds_ptr,
    masked_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    dropped_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    SEED_INDEX_CONST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    query = rows - flat_bh * Q_LEN
    offsets = rows[:, None] * K_LEN + cols[None, :]
    mask_offsets = (
        batch[:, None] * (Q_LEN * K_LEN)
        + query[:, None] * K_LEN
        + cols[None, :]
    )

    x = tl.load(x_ptr + offsets, mask=active, other=0.0)
    mask = tl.load(mask_ptr + mask_offsets, mask=active, other=0).to(tl.int1)
    fill = tl.load(fill_ptr)
    masked = tl.where(mask, fill, x)
    tl.store(masked_ptr + offsets, masked, mask=active)

    scores = tl.where(active, masked.to(tl.float32), -float("inf"))
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    seed = tl.load(seeds_ptr + SEED_INDEX_CONST)
    random = tl.rand(seed, offsets.to(tl.uint32))
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32)
    keep = random > dropout_p
    tl.store(keep_ptr + offsets, keep, mask=active)

    dropped = tl.where(keep, probs, 0.0)
    scaled = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled, mask=active)


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


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, _shape2 = inputs
    del _shape0, _shape1, _shape2

    batch = 8
    heads = 24
    q_len = 512
    k_len = 512
    rows = batch * heads * q_len
    full_shape = (batch, heads, q_len, k_len)
    row_shape = (batch, heads, q_len, 1)
    out_shape = (batch * heads, q_len, k_len)

    masked = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape,
        (heads * q_len, q_len, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        (heads * q_len, q_len, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape,
        (q_len * k_len, k_len, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _masked_softmax_dropout_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            masked,
            amax,
            sum_1,
            keep,
            dropped,
            ROWS=rows,
            HEADS=heads,
            Q_LEN=q_len,
            K_LEN=k_len,
            SEED_INDEX_CONST=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            full_shape,
            seed,
            device=arg0_1.device,
        )
        _masked_softmax_dropout_random_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            random,
            masked,
            amax,
            sum_1,
            keep,
            dropped,
            ROWS=rows,
            HEADS=heads,
            Q_LEN=q_len,
            K_LEN=k_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return masked, amax, sum_1, keep, dropped, dropped.permute(0, 2, 1)


# 00541467: DeBERTa bf16 masked attention softmax/dropout, B=8, H=24, S=512.
@oracle_impl(hardware="B200", point="00541467", BLOCK_M=2, BLOCK_N=512, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
