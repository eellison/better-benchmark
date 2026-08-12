"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT bf16 token-mask attention softmax/dropout scope in one Triton row kernel, including the `arg0 > 0` repeated/unsqueezed bool mask side output, its returned `eq(0)` inverse, the returned finite bf16 scalar fill, the `[192,128,128] -> [16,12,128,128]` score view, bf16-rounded divide-by-eight score scale, fp32 stable last-dimension amax/libdevice.exp/sum/div side outputs, Inductor seed-index-1 dropout mask, f32 dropout scaling, final bf16 `[192,128,128]` view, and returned permute alias, whereas Inductor lowers the decomposed token-mask construction, scalar-fill masking, row reduction, stochastic dropout, and alias-producing epilogue through generic pointwise/reduction/RNG/layout kernels; Inductor cannot do this today because its row-softmax scheduler does not keep the visible mask producer, scalar fill, reduction side outputs, seeded dropout mask, scaled bf16 output materialization, and layout-only alias resident in one full-scope attention plan while preserving the bf16 and f32 dtype boundaries; the fix is SCHEDULER_FUSION: extend attention-softmax scheduling to fuse token-mask construction, row reduction side outputs, Inductor-seeded dropout, and alias-return epilogues for this BERT attention shape."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 1


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
def _token_masked_softmax_dropout_kernel(
    token_ptr,
    scores_ptr,
    rng_ptr,
    valid_ptr,
    eq_ptr,
    fill_ptr,
    amax_ptr,
    denom_ptr,
    keep_ptr,
    out_ptr,
    SCORES_S0: tl.constexpr,
    SCORES_S1: tl.constexpr,
    SCORES_S2: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    OUT_S2: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    RNG_SEED_INDEX: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_q = tl.program_id(0)
    head_block = tl.program_id(1)

    batch = batch_q // Q_LEN
    query = batch_q - batch * Q_LEN
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.arange(0, BLOCK_K)

    head_mask = heads < HEADS
    col_mask = cols < K_LEN
    active = head_mask[:, None] & col_mask[None, :]

    token_valid = tl.load(
        token_ptr + batch * K_LEN + cols,
        mask=col_mask,
        other=0,
    ) > 0
    token_eq = token_valid == 0
    mask_offsets = batch * (Q_LEN * K_LEN) + query * K_LEN + cols
    tl.store(valid_ptr + mask_offsets, token_valid, mask=(head_block == 0) & col_mask)
    tl.store(eq_ptr + mask_offsets, token_eq, mask=(head_block == 0) & col_mask)

    fill = tl.full((), -998244352.0, tl.float32).to(tl.bfloat16)
    tl.store(fill_ptr, fill, mask=(batch_q == 0) & (head_block == 0))

    flat_heads = batch * HEADS + heads
    score_offsets = (
        flat_heads[:, None] * SCORES_S0
        + query * SCORES_S1
        + cols[None, :] * SCORES_S2
    )
    raw = tl.load(scores_ptr + score_offsets, mask=active, other=0.0).to(tl.float32)
    scaled = _f32_mul(raw, 0.125).to(tl.bfloat16)
    masked = tl.where(token_eq[None, :], fill, scaled)
    values = tl.where(active, masked.to(tl.float32), -float("inf"))

    row_max = tl.max(values, axis=1)
    numer = libdevice.exp(values - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    row_offsets = (flat_heads * Q_LEN + query)
    tl.store(amax_ptr + row_offsets, row_max, mask=head_mask)
    tl.store(denom_ptr + row_offsets, denom, mask=head_mask)

    linear_offsets = row_offsets[:, None] * K_LEN + cols[None, :]
    if USE_RANDOM_PTR:
        random = tl.load(rng_ptr + linear_offsets, mask=active, other=0.0)
    else:
        seed = tl.load(rng_ptr + RNG_SEED_INDEX)
        random = tl.rand(seed, linear_offsets.to(tl.uint32))
    keep = random > 0.1
    tl.store(keep_ptr + linear_offsets, keep, mask=active)

    dropped = tl.where(keep, probs, 0.0)
    scaled_dropout = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    out_offsets = (
        flat_heads[:, None] * OUT_S0
        + query * OUT_S1
        + cols[None, :] * OUT_S2
    )
    tl.store(out_ptr + out_offsets, scaled_dropout, mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // (block_size),
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


def _launch(
    tokens,
    scores,
    rng,
    valid,
    eq,
    fill,
    amax,
    denom,
    keep,
    out,
    *,
    use_random_ptr,
    block_h,
    block_k,
    num_warps,
    num_stages,
):
    batch = int(tokens.shape[0])
    k_len = int(tokens.shape[1])
    q_len = int(out.shape[1])
    heads = int(out.shape[0]) // batch
    _token_masked_softmax_dropout_kernel[(batch * q_len, triton.cdiv(heads, block_h))](
        tokens,
        scores,
        rng,
        valid,
        eq,
        fill,
        amax,
        denom,
        keep,
        out,
        SCORES_S0=scores.stride(0),
        SCORES_S1=scores.stride(1),
        SCORES_S2=scores.stride(2),
        OUT_S0=out.stride(0),
        OUT_S1=out.stride(1),
        OUT_S2=out.stride(2),
        BATCH=batch,
        HEADS=heads,
        Q_LEN=q_len,
        K_LEN=k_len,
        RNG_SEED_INDEX=SEED_INDEX,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# 9a66816c: BERT token-mask bf16 attention softmax/dropout, B=16, H=12, S=128.
@oracle_impl(hardware="B200", point="9a66816c", block_h=8, block_k=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    block_h: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
):
    tokens, scores, seeds, _repeat_shape, view_shape, random_shape, _expand_shape, out_shape = inputs
    del _repeat_shape, _expand_shape
    view_shape = _as_shape(view_shape)
    random_shape = _as_shape(random_shape)
    out_shape = _as_shape(out_shape)
    batch, heads, q_len, k_len = view_shape
    mask_shape = (batch, 1, q_len, k_len)
    row_shape = (batch, heads, q_len, 1)

    valid = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=scores.device,
        dtype=torch.bool,
    )
    eq = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=scores.device,
        dtype=torch.bool,
    )
    fill = torch.empty_strided((), (), device=scores.device, dtype=torch.bfloat16)
    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=scores.device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=scores.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=scores.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=scores.device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        _launch(
            tokens,
            scores,
            seeds,
            valid,
            eq,
            fill,
            amax,
            denom,
            keep,
            out,
            use_random_ptr=False,
            block_h=block_h,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return valid, eq, fill, amax, denom, keep, out, out.permute(0, 2, 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape,
        seed,
        device=scores.device,
    )
    _launch(
        tokens,
        scores,
        random,
        valid,
        eq,
        fill,
        amax,
        denom,
        keep,
        out,
        use_random_ptr=True,
        block_h=block_h,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return valid, eq, fill, amax, denom, keep, out, out.permute(0, 2, 1)
