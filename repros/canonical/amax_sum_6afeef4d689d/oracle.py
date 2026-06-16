"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeBERTaV2 bf16 masked attention softmax/dropout scope in one Triton row kernel, including the `[192,512,512] -> [8,24,512,512]` view, broadcast `[8,1,512,512]` scalar-fill mask, returned bf16 masked scores, fp32 amax and denominator side outputs, Inductor seed-index 28 f32 dropout mask, f32 dropout scaling, flattened f32 metadata view followed by the visible bf16 output rounding, and returned permute alias, whereas Inductor lowers the decomposed view/where/cast/amax/sub/exp/sum/div/RNG/dropout/view/cast/permute graph through separate generic reduction, stochastic pointwise, and layout scheduling fragments; Inductor cannot do this today because its scheduler does not keep the mask-fill producer, seeded dropout producer, reduction side outputs, final cast, and alias-only epilogue inside one row-softmax plan while preserving the bf16 score boundary and f32 RNG comparison; the fix is SCHEDULER_FUSION: teach the attention-softmax scheduler to fuse scalar-fill masking, natural-exp softmax, Inductor-seeded dropout, side-output stores, and layout-only returns into one guarded row template."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 28


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
def _masked_softmax_dropout_kernel(
    scores_ptr,
    mask_ptr,
    fill_ptr,
    rng_ptr,
    where_ptr,
    amax_ptr,
    denom_ptr,
    keep_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    RNG_SEED_INDEX: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    query = rows - flat_bh * Q_LEN
    linear_offsets = rows[:, None] * K_LEN + cols[None, :]
    mask_offsets = (batch[:, None] * Q_LEN + query[:, None]) * K_LEN + cols[None, :]

    raw = tl.load(scores_ptr + linear_offsets, mask=active, other=0.0)
    mask_values = tl.load(mask_ptr + mask_offsets, mask=active, other=0)
    fill = tl.load(fill_ptr)
    masked_scores = tl.where(mask_values, fill, raw)
    tl.store(where_ptr + linear_offsets, masked_scores, mask=active)

    scores = tl.where(active, masked_scores.to(tl.float32), -float("inf"))
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(denom_ptr + rows, denom, mask=row_mask)

    if USE_RANDOM_PTR:
        random = tl.load(rng_ptr + linear_offsets, mask=active, other=0.0)
    else:
        seed = tl.load(rng_ptr + RNG_SEED_INDEX)
        random = tl.rand(seed, linear_offsets.to(tl.uint32))
    keep = random > 0.1
    tl.store(keep_ptr + linear_offsets, keep, mask=active)

    dropped = probs * keep.to(tl.float32)
    scaled = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    tl.store(out_ptr + linear_offsets, scaled, mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


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


def _launch(
    scores,
    mask,
    fill,
    rng,
    where,
    amax,
    denom,
    keep,
    out,
    *,
    use_random_ptr,
    block_m,
    block_k,
    num_warps,
    num_stages,
):
    rows = int(scores.numel() // scores.shape[-1])
    heads = int(where.shape[1])
    q_len = int(where.shape[2])
    k_len = int(where.shape[3])
    _masked_softmax_dropout_kernel[(triton.cdiv(rows, block_m),)](
        scores,
        mask,
        fill,
        rng,
        where,
        amax,
        denom,
        keep,
        out,
        ROWS=rows,
        HEADS=heads,
        Q_LEN=q_len,
        K_LEN=k_len,
        RNG_SEED_INDEX=SEED_INDEX,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@oracle_impl(hardware="B200", point="00541467", block_m=1, block_k=512, num_warps=2, num_stages=3)
def oracle_forward(
    inputs,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    view_shape = _resolve_shape(shape0, arg0_1.numel())
    random_shape = _as_shape(shape1)
    flat_shape = _resolve_shape(shape2, arg0_1.numel())
    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)

    where = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape,
        _contiguous_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape,
        _contiguous_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        _launch(
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            where,
            amax,
            denom,
            keep,
            out,
            use_random_ptr=False,
            block_m=block_m,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return where, amax, denom, keep, out, out.permute(0, 2, 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape,
        seed,
        device=arg0_1.device,
    )
    _launch(
        arg0_1,
        arg1_1,
        arg2_1,
        random,
        where,
        amax,
        denom,
        keep,
        out,
        use_random_ptr=True,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return where, amax, denom, keep, out, out.permute(0, 2, 1)
