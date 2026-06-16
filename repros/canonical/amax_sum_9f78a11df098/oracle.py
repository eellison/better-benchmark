"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT bf16 scalar-fill masked attention softmax/dropout scope from Repro.forward in one row kernel, including the `[192, 128, 128]` to `[16, 12, 128, 128]` view, bf16 divide-by-eight score rounding, broadcast bool mask scalar-fill and returned bf16 masked scores, fp32 stable last-dimension amax/libdevice.exp/sum/div side outputs, Inductor seed-index-41 f32 dropout RNG with the returned bool mask, f32 dropout scaling, final bf16 `[192, 128, 128]` view, and returned permute alias, whereas Inductor lowers the mask producer, reduction, stochastic dropout, and alias-producing epilogue through generic pointwise/reduction/RNG/layout kernels; Inductor cannot do this today because its row-softmax scheduler does not keep the scalar-fill mask, observable reduction side outputs, seeded dropout mask generation, scaled bf16 output materialization, and layout-only alias resident in one full-scope plan while preserving the bf16 and f32 dtype boundaries; the fix is SCHEDULER_FUSION: extend attention-softmax scheduling to fuse scalar-fill masking, row reduction side outputs, Inductor-seeded dropout, and alias-return epilogues for this BERT attention shape."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 41


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
    random_or_seed_ptr,
    where_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    mask_s0: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    n_rows: tl.constexpr,
    n_heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    seed_index: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    bh = rows // q_len
    q = rows - bh * q_len
    b = bh // n_heads

    dense_offsets = rows[:, None] * k_len + cols[None, :]
    mask_offsets = b[:, None] * mask_s0 + q[:, None] * mask_s2 + cols[None, :] * mask_s3

    score = tl.load(scores_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    scaled_bf16 = _f32_mul(score, 0.125).to(tl.bfloat16)
    fill = tl.load(fill_ptr).to(tl.bfloat16)
    mask_fill = tl.load(mask_ptr + mask_offsets, mask=mask, other=0)
    rounded = tl.where(mask_fill, fill, scaled_bf16)
    tl.store(where_ptr + dense_offsets, rounded, mask=mask)

    values = tl.where(mask, rounded.to(tl.float32), -float("inf"))
    row_max = tl.max(values, axis=1)
    numer = libdevice.exp(values - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    if USE_RANDOM_PTR:
        random = tl.load(
            random_or_seed_ptr + dense_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed = tl.load(random_or_seed_ptr + seed_index)
        random = tl.rand(seed, dense_offsets.to(tl.uint32))

    keep = random > 0.1
    tl.store(gt_ptr + dense_offsets, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0)
    scaled = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + dense_offsets, scaled, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
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


# 0e2c5e9e: BERT train bf16 scores [192,128,128], mask [16,1,128,128], seed-index 41.
@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    scores, mask, fill, seeds, full_shape, random_shape, _expand_shape, out_shape = inputs
    full_shape = _shape_tuple(full_shape)
    random_shape = _shape_tuple(random_shape)
    out_shape = _shape_tuple(out_shape)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(scores.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    where = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=scores.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=scores.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=scores.device,
        dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=scores.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=scores.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _masked_softmax_dropout_kernel[grid](
            scores,
            mask,
            fill,
            seeds,
            where,
            amax,
            sum_1,
            gt,
            dropped,
            mask_s0=mask.stride(0),
            mask_s2=mask.stride(2),
            mask_s3=mask.stride(3),
            n_rows=n_rows,
            n_heads=n_heads,
            q_len=q_len,
            k_len=k_len,
            seed_index=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            USE_RANDOM_PTR=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=scores.device,
        )
        _masked_softmax_dropout_kernel[grid](
            scores,
            mask,
            fill,
            random,
            where,
            amax,
            sum_1,
            gt,
            dropped,
            mask_s0=mask.stride(0),
            mask_s2=mask.stride(2),
            mask_s3=mask.stride(3),
            n_rows=n_rows,
            n_heads=n_heads,
            q_len=q_len,
            k_len=k_len,
            seed_index=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            USE_RANDOM_PTR=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return where, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
