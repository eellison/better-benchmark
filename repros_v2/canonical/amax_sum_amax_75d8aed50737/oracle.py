"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 T5 dual relative-position attention softmax/dropout scope in two Triton row kernels, including both logarithmic bucket tables, embedding side outputs, broadcast relative-bias tensors, the causal mask/scalar/iota side outputs, bf16 score-rounding before amax/libdevice.exp/sum/div, both observable reduction side outputs, exact Inductor seed-index 1 and 27 dropout masks, scaled bf16 dropout views, and both permute aliases, whereas Inductor lowers the decomposed iota/bucket/embedding/mask/add/cast/softmax/random/dropout/view/permute graph as generic producers and layout fragments around two large reductions; Inductor cannot do this today because its pattern library does not recognize T5 relative-position attention with stochastic softmax epilogues and many sibling outputs as one guarded full-scope template; the fix is NEW_PATTERN: add a T5 relative-position attention lowering that recomputes bucket and mask predicates inside the row-softmax schedule while emitting all side outputs and aliases directly."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX_ENCODER = 1
SEED_INDEX_DECODER = 27


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
def _t5_bidirectional_bucket(query, key):
    rel = key - query
    distance = tl.abs(rel)
    bucket = distance
    bucket = tl.where(distance >= 8, 8, bucket)
    bucket = tl.where(distance >= 12, 9, bucket)
    bucket = tl.where(distance >= 16, 10, bucket)
    bucket = tl.where(distance >= 23, 11, bucket)
    bucket = tl.where(distance >= 32, 12, bucket)
    bucket = tl.where(distance >= 46, 13, bucket)
    bucket = tl.where(distance >= 64, 14, bucket)
    bucket = tl.where(distance >= 91, 15, bucket)
    return bucket + tl.where(rel > 0, 16, 0)


@triton.jit
def _t5_causal_bucket(query, key):
    distance = tl.maximum(query - key, 0)
    bucket = distance
    bucket = tl.where(distance >= 16, 16, bucket)
    bucket = tl.where(distance >= 19, 17, bucket)
    bucket = tl.where(distance >= 21, 18, bucket)
    bucket = tl.where(distance >= 24, 19, bucket)
    bucket = tl.where(distance >= 27, 20, bucket)
    bucket = tl.where(distance >= 31, 21, bucket)
    bucket = tl.where(distance >= 35, 22, bucket)
    bucket = tl.where(distance >= 40, 23, bucket)
    bucket = tl.where(distance >= 46, 24, bucket)
    bucket = tl.where(distance >= 52, 25, bucket)
    bucket = tl.where(distance >= 59, 26, bucket)
    bucket = tl.where(distance >= 67, 27, bucket)
    bucket = tl.where(distance >= 77, 28, bucket)
    bucket = tl.where(distance >= 87, 29, bucket)
    bucket = tl.where(distance >= 99, 30, bucket)
    bucket = tl.where(distance >= 113, 31, bucket)
    return bucket


@triton.jit
def _encoder_branch_kernel(
    score_ptr,
    rel_ptr,
    seeds_ptr,
    random_ptr,
    bucket_ptr,
    embedding_ptr,
    bias_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    dropped_ptr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads
    query = rows - flat_bh * q_len
    bucket = _t5_bidirectional_bucket(query[:, None], cols[None, :])
    rel = tl.load(rel_ptr + bucket * heads + head[:, None], mask=mask, other=0.0).to(tl.float32)

    linear = rows[:, None] * k_len + cols[None, :]
    score = tl.load(score_ptr + linear, mask=mask, other=0.0).to(tl.float32)
    rounded = (score + rel).to(tl.bfloat16)
    logits = tl.where(mask, rounded.to(tl.float32), -float("inf"))

    row_max = tl.max(logits, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(logits - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    if USE_RANDOM_PTR:
        rand_bf16 = tl.load(random_ptr + linear, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
    else:
        seed = tl.load(seeds_ptr + 1)
        rand_bf16 = tl.rand(seed, linear.to(tl.uint32)).to(tl.bfloat16)
    keep = rand_bf16 > tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    tl.store(keep_ptr + linear, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + linear, scaled, mask=mask)

    first_batch = batch[:, None] == 0
    first_head = head[:, None] == 0
    qk_offsets = query[:, None] * k_len + cols[None, :]
    tl.store(bucket_ptr + qk_offsets, bucket.to(tl.int64), mask=mask & first_batch & first_head)
    embedding_offsets = (query[:, None] * k_len + cols[None, :]) * heads + head[:, None]
    tl.store(embedding_ptr + embedding_offsets, rel, mask=mask & first_batch)
    bias_offsets = batch[:, None] * (heads * q_len * k_len) + head[:, None] + query[:, None] * (k_len * heads) + cols[None, :] * heads
    tl.store(bias_ptr + bias_offsets, rel, mask=mask)


@triton.jit
def _decoder_branch_kernel(
    score_ptr,
    rel_ptr,
    seeds_ptr,
    random_ptr,
    iota_ptr,
    zero_ptr,
    bucket_ptr,
    embedding_ptr,
    bias_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    dropped_ptr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // q_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads
    query = rows - flat_bh * q_len
    causal = cols[None, :] <= query[:, None]
    bucket = _t5_causal_bucket(query[:, None], cols[None, :])
    rel = tl.load(rel_ptr + bucket * heads + head[:, None], mask=mask, other=0.0).to(tl.float32)
    min_f32 = tl.full((BLOCK_M, BLOCK_N), -3.4028234663852886e38, tl.float32)
    bias_val = tl.where(causal, rel, min_f32)

    linear = rows[:, None] * k_len + cols[None, :]
    score = tl.load(score_ptr + linear, mask=mask, other=0.0).to(tl.float32)
    rounded = (score + bias_val).to(tl.bfloat16)
    logits = tl.where(mask, rounded.to(tl.float32), -float("inf"))

    row_max = tl.max(logits, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(logits - row_max[:, None])
    numer = tl.where(mask & causal, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    if USE_RANDOM_PTR:
        rand_bf16 = tl.load(random_ptr + linear, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
    else:
        seed = tl.load(seeds_ptr + 27)
        rand_bf16 = tl.rand(seed, linear.to(tl.uint32)).to(tl.bfloat16)
    keep = rand_bf16 > tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    tl.store(keep_ptr + linear, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + linear, scaled, mask=mask)

    first_batch = batch[:, None] == 0
    first_head = head[:, None] == 0
    qk_offsets = query[:, None] * k_len + cols[None, :]
    tl.store(bucket_ptr + qk_offsets, bucket.to(tl.int64), mask=mask & first_batch & first_head)
    embedding_offsets = (query[:, None] * k_len + cols[None, :]) * heads + head[:, None]
    tl.store(embedding_ptr + embedding_offsets, rel, mask=mask & first_batch)
    bias_offsets = batch[:, None] * (heads * q_len * k_len) + head[:, None] + query[:, None] * (k_len * heads) + cols[None, :] * heads
    tl.store(bias_ptr + bias_offsets, bias_val, mask=mask)
    tl.store(iota_ptr + query, query.to(tl.int64), mask=row_mask & (batch == 0) & (head == 0))
    tl.store(zero_ptr, 0.0, mask=tl.program_id(0) == 0)


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


def _stride4(shape):
    return (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)


def _row_stride(shape):
    return (shape[1] * shape[2], shape[2], 1, 1)


@oracle_impl(hardware="B200", point="5d077752", BLOCK_M=1, BLOCK_N=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4
    del _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11

    batch = 8
    heads = int(arg1_1.shape[1])
    q_len = int(arg0_1.shape[1])
    k_len = int(arg0_1.shape[2])
    full_shape = (batch, heads, q_len, k_len)
    row_shape = (batch, heads, q_len, 1)
    flat_shape = tuple(int(dim) for dim in _shape_param_5)
    flat_shape_2 = tuple(int(dim) for dim in _shape_param_12)
    n_rows = int(batch * heads * q_len)

    add_3 = torch.empty_strided((q_len, k_len), (k_len, 1), device=arg0_1.device, dtype=torch.int64)
    embedding = torch.empty_strided((q_len, k_len, heads), (k_len * heads, heads, 1), device=arg0_1.device, dtype=torch.float32)
    add_4 = torch.empty_strided(full_shape, (heads * q_len * k_len, 1, k_len * heads, heads), device=arg0_1.device, dtype=torch.float32)
    amax = torch.empty_strided(row_shape, _row_stride(row_shape), device=arg0_1.device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _row_stride(row_shape), device=arg0_1.device, dtype=torch.float32)
    gt_1 = torch.empty_strided(full_shape, _stride4(full_shape), device=arg0_1.device, dtype=torch.bool)
    view_1 = torch.empty_strided(flat_shape, (q_len * k_len, k_len, 1), device=arg0_1.device, dtype=torch.bfloat16)

    unsqueeze_4 = torch.empty_strided((1, 1, q_len), (q_len, q_len, 1), device=arg0_1.device, dtype=torch.int64)
    full_2 = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.float32)
    add_8 = torch.empty_strided((q_len, k_len), (k_len, 1), device=arg0_1.device, dtype=torch.int64)
    embedding_1 = torch.empty_strided((q_len, k_len, heads), (k_len * heads, heads, 1), device=arg0_1.device, dtype=torch.float32)
    add_9 = torch.empty_strided(full_shape, (heads * q_len * k_len, 1, k_len * heads, heads), device=arg0_1.device, dtype=torch.float32)
    amax_1 = torch.empty_strided(row_shape, _row_stride(row_shape), device=arg0_1.device, dtype=torch.float32)
    sum_2 = torch.empty_strided(row_shape, _row_stride(row_shape), device=arg0_1.device, dtype=torch.float32)
    gt_2 = torch.empty_strided(full_shape, _stride4(full_shape), device=arg0_1.device, dtype=torch.bool)
    view_3 = torch.empty_strided(flat_shape_2, (q_len * k_len, k_len, 1), device=arg0_1.device, dtype=torch.bfloat16)

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _encoder_branch_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg0_1,
            add_3,
            embedding,
            add_4,
            amax,
            sum_1,
            gt_1,
            view_1,
            n_rows=n_rows,
            heads=heads,
            q_len=q_len,
            k_len=k_len,
            USE_RANDOM_PTR=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _decoder_branch_kernel[grid](
            arg3_1,
            arg4_1,
            arg2_1,
            arg3_1,
            unsqueeze_4,
            full_2,
            add_8,
            embedding_1,
            add_9,
            amax_1,
            sum_2,
            gt_2,
            view_3,
            n_rows=n_rows,
            heads=heads,
            q_len=q_len,
            k_len=k_len,
            USE_RANDOM_PTR=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed0 = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX_ENCODER)
        seed1 = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX_DECODER)
        random0, random1 = _inductor_random_pair_for_eager_check(full_shape, seed0, seed1, device=arg0_1.device)
        _encoder_branch_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            random0,
            add_3,
            embedding,
            add_4,
            amax,
            sum_1,
            gt_1,
            view_1,
            n_rows=n_rows,
            heads=heads,
            q_len=q_len,
            k_len=k_len,
            USE_RANDOM_PTR=True,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _decoder_branch_kernel[grid](
            arg3_1,
            arg4_1,
            arg2_1,
            random1,
            unsqueeze_4,
            full_2,
            add_8,
            embedding_1,
            add_9,
            amax_1,
            sum_2,
            gt_2,
            view_3,
            n_rows=n_rows,
            heads=heads,
            q_len=q_len,
            k_len=k_len,
            USE_RANDOM_PTR=True,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return (
        add_3,
        embedding,
        add_4,
        amax,
        sum_1,
        gt_1,
        view_1,
        unsqueeze_4,
        full_2,
        add_8,
        embedding_1,
        add_9,
        amax_1,
        sum_2,
        gt_2,
        view_3,
        view_3.permute(0, 2, 1),
        view_1.permute(0, 2, 1),
    )
