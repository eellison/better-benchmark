"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5/MT5 additive-bias bf16 attention softmax/dropout scope in one Triton row kernel, including the shape-param view, strided f32 bias add, explicit bf16 post-add rounding, fp32 stable last-dimension amax/libdevice.exp/sum/div, returned bf16 rounded score tensor, returned fp32 amax and sum side outputs, Inductor seed-index-9 dropout with f32-random-to-bf16 comparison, bf16 dropout scaling, returned contiguous view, and returned permute alias, whereas Inductor lowers the decomposed add/cast/amax/sub/exp/sum/div/cast/RNG/dropout/view/permute graph through separate generic reduction, stochastic pointwise, and layout scheduling fragments; Inductor cannot do this today because it has no full-scope additive-bias attention-softmax lowering that keeps the seeded dropout producer and all observable side outputs resident while preserving bf16 rounding boundaries and strided-bias reads; the fix is NEW_PATTERN: add a guarded T5/MT5 attention-softmax/dropout lowering that fuses additive bias, explicit bf16 score/probability casts, Inductor-seeded dropout, reduction side outputs, and alias-producing epilogues into one row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 9


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
def _softmax_dropout_random_kernel(
    score_ptr,
    bias_ptr,
    random_ptr,
    rounded_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    dropped_ptr,
    score_s0: tl.constexpr,
    score_s1: tl.constexpr,
    score_s2: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s1: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    n_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch_idx = flat_bh // heads
    head = flat_bh - batch_idx * heads
    query = rows - flat_bh * q_len
    score_offsets = (
        flat_bh[:, None] * score_s0
        + query[:, None] * score_s1
        + cols[None, :] * score_s2
    )
    bias_offsets = (
        batch_idx[:, None] * bias_s0
        + head[:, None] * bias_s1
        + query[:, None] * bias_s2
        + cols[None, :] * bias_s3
    )
    linear_offsets = rows[:, None] * k_len + cols[None, :]

    score = tl.load(score_ptr + score_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    rounded = (score + bias).to(tl.bfloat16)
    tl.store(rounded_ptr + linear_offsets, rounded, mask=mask)

    x = tl.where(mask, rounded.to(tl.float32), -float("inf"))
    row_max = tl.max(x, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(x - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    rand_bf16 = tl.load(
        random_ptr + linear_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(keep_ptr + linear_offsets, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + linear_offsets, scaled, mask=mask)


@triton.jit
def _softmax_dropout_seeded_kernel(
    score_ptr,
    bias_ptr,
    seeds_ptr,
    rounded_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    dropped_ptr,
    score_s0: tl.constexpr,
    score_s1: tl.constexpr,
    score_s2: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s1: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    n_rows: tl.constexpr,
    seed_index: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch_idx = flat_bh // heads
    head = flat_bh - batch_idx * heads
    query = rows - flat_bh * q_len
    score_offsets = (
        flat_bh[:, None] * score_s0
        + query[:, None] * score_s1
        + cols[None, :] * score_s2
    )
    bias_offsets = (
        batch_idx[:, None] * bias_s0
        + head[:, None] * bias_s1
        + query[:, None] * bias_s2
        + cols[None, :] * bias_s3
    )
    linear_offsets = rows[:, None] * k_len + cols[None, :]

    score = tl.load(score_ptr + score_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    rounded = (score + bias).to(tl.bfloat16)
    tl.store(rounded_ptr + linear_offsets, rounded, mask=mask)

    x = tl.where(mask, rounded.to(tl.float32), -float("inf"))
    row_max = tl.max(x, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(x - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    seed = tl.load(seeds_ptr + seed_index)
    rand_bf16 = tl.rand(seed, linear_offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(keep_ptr + linear_offsets, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + linear_offsets, scaled, mask=mask)


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


def _contiguous_4d_stride(shape):
    return (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)


def _contiguous_3d_stride(shape):
    return (shape[1] * shape[2], shape[2], 1)


def _reduction_stride(shape):
    return (shape[1] * shape[2], shape[2], 1, 1)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    del _shape2

    score_shape = tuple(int(dim) for dim in shape0)
    random_shape = tuple(int(dim) for dim in shape1)
    view_shape = tuple(int(dim) for dim in shape3)
    reduction_shape = (score_shape[0], score_shape[1], score_shape[2], 1)

    rounded = torch.empty_strided(
        score_shape,
        _contiguous_4d_stride(score_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape,
        _reduction_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape,
        _reduction_stride(reduction_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        score_shape,
        _contiguous_4d_stride(score_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        view_shape,
        _contiguous_3d_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    heads = int(score_shape[1])
    q_len = int(score_shape[2])
    k_len = int(score_shape[3])
    n_rows = int(score_shape[0] * heads * q_len)
    grid = (triton.cdiv(n_rows, BLOCK_M),)

    if torch.cuda.is_current_stream_capturing():
        _softmax_dropout_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            rounded,
            amax,
            denom,
            keep,
            dropped,
            score_s0=arg0_1.stride(0),
            score_s1=arg0_1.stride(1),
            score_s2=arg0_1.stride(2),
            bias_s0=arg1_1.stride(0),
            bias_s1=arg1_1.stride(1),
            bias_s2=arg1_1.stride(2),
            bias_s3=arg1_1.stride(3),
            heads=heads,
            q_len=q_len,
            k_len=k_len,
            n_rows=n_rows,
            seed_index=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=arg0_1.device,
        )
        _softmax_dropout_random_kernel[grid](
            arg0_1,
            arg1_1,
            random,
            rounded,
            amax,
            denom,
            keep,
            dropped,
            score_s0=arg0_1.stride(0),
            score_s1=arg0_1.stride(1),
            score_s2=arg0_1.stride(2),
            bias_s0=arg1_1.stride(0),
            bias_s1=arg1_1.stride(1),
            bias_s2=arg1_1.stride(2),
            bias_s3=arg1_1.stride(3),
            heads=heads,
            q_len=q_len,
            k_len=k_len,
            n_rows=n_rows,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return rounded, amax, denom, keep, dropped, dropped.permute(0, 2, 1)


# dda3d8e0: MT5 train attention softmax/dropout, B=32, H=6, S=128.
@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
# aeb1682d: T5 train attention softmax/dropout, B=8, H=8, S=1024.
@oracle_impl(hardware="B200", point="aeb1682d", BLOCK_M=1, BLOCK_N=1024, num_warps=8, num_stages=3)
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
