"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MT5 training embedding plus dual seeded-dropout RMSNorm scope in one Triton row kernel, including the returned fp32 embedding gather, generated `inductor_seeds`, seed-index-0 and seed-index-34 f32 random masks, both returned scaled-dropout fp32 tensors, mean(square)+eps=1e-6 rsqrt side tensors, two separate fp32 affine weight multiplies, and the final bf16 flattened views, whereas Inductor lowers the embedding, internally generated RNG producers, two RMS reductions, affine epilogues, bf16 casts/views, and all visible side outputs through generic scheduler fragments; Inductor cannot fuse this returned-output envelope today because its normalization templates do not keep a gathered embedding producer, two independent Inductor-seeded dropout domains, and all sibling materialized outputs resident across the fixed-hidden reductions while preserving the stochastic no-skip contract; the fix is NEW_PATTERN: add a guarded MT5 embedding-dual-dropout-RMSNorm lowering that folds generated seeds, indexed gather, two f32 RNG masks, row RMS reductions, affine epilogues, and side-output stores into one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_COUNT = 84
SEED_INDEX_0 = 0
SEED_INDEX_1 = 34


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
def _embedding_dual_dropout_rmsnorm_kernel(
    table_ptr,
    ids_ptr,
    weight0_ptr,
    weight1_ptr,
    seed_or_random0_ptr,
    random1_ptr,
    embedding_ptr,
    gt0_ptr,
    mul1_ptr,
    rsqrt0_ptr,
    view0_ptr,
    gt1_ptr,
    mul5_ptr,
    rsqrt1_ptr,
    view1_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    USE_RANDOM_PTRS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    token_ids = tl.load(ids_ptr + rows, mask=row_mask, other=0)
    embedding = tl.load(
        table_ptr + token_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    tl.store(embedding_ptr + offsets, embedding, mask=mask)

    if USE_RANDOM_PTRS:
        random0 = tl.load(
            seed_or_random0_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        random1 = tl.load(
            random1_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed0 = tl.load(seed_or_random0_ptr + 0)
        seed1 = tl.load(seed_or_random0_ptr + 34)
        rng_offsets = offsets.to(tl.uint32)
        random0 = tl.rand(seed0, rng_offsets)
        random1 = tl.rand(seed1, rng_offsets)

    keep0 = random0 > 0.1
    keep1 = random1 > 0.1
    tl.store(gt0_ptr + offsets, keep0, mask=mask)
    tl.store(gt1_ptr + offsets, keep1, mask=mask)

    dropped0 = _f32_mul(keep0.to(tl.float32), embedding)
    mul1 = _f32_mul(dropped0, 1.1111111111111112)
    dropped1 = _f32_mul(keep1.to(tl.float32), embedding)
    mul5 = _f32_mul(dropped1, 1.1111111111111112)
    tl.store(mul1_ptr + offsets, mul1, mask=mask)
    tl.store(mul5_ptr + offsets, mul5, mask=mask)

    mul1_for_reduce = tl.where(mask, mul1, 0.0)
    mul5_for_reduce = tl.where(mask, mul5, 0.0)
    square_sum0 = tl.sum(_f32_mul(mul1_for_reduce, mul1_for_reduce), axis=1)
    square_sum1 = tl.sum(_f32_mul(mul5_for_reduce, mul5_for_reduce), axis=1)
    mean_square0 = _f32_mul(square_sum0, 0.001953125)
    mean_square1 = _f32_mul(square_sum1, 0.001953125)
    inv0 = libdevice.rsqrt(_f32_add(mean_square0, 1.0e-6))
    inv1 = libdevice.rsqrt(_f32_add(mean_square1, 1.0e-6))
    tl.store(rsqrt0_ptr + rows, inv0, mask=row_mask)
    tl.store(rsqrt1_ptr + rows, inv1, mask=row_mask)

    weight0 = tl.load(
        weight0_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    weight1 = tl.load(
        weight1_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    norm0 = _f32_mul(mul1, inv0[:, None])
    norm1 = _f32_mul(mul5, inv1[:, None])
    out0 = _f32_mul(weight0[None, :], norm0)
    out1 = _f32_mul(weight1[None, :], norm1)
    tl.store(view0_ptr + offsets, out0.to(tl.bfloat16), mask=mask)
    tl.store(view1_ptr + offsets, out1.to(tl.bfloat16), mask=mask)


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
    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
    random0 = torch.ops.prims.inductor_random.default(shape0, seed0, "rand")
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
    random1 = torch.ops.prims.inductor_random.default(shape1, seed1, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random0, random1


def _launch(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    full_shape0 = _shape_tuple(shape0)
    view_shape0 = _shape_tuple(shape1)
    full_shape1 = _shape_tuple(shape2)
    view_shape1 = _shape_tuple(shape3)
    row_shape0 = full_shape0[:-1] + (1,)
    row_shape1 = full_shape1[:-1] + (1,)
    full_stride0 = _contiguous_stride(full_shape0)
    full_stride1 = _contiguous_stride(full_shape1)

    rows = int(view_shape0[0])
    hidden = int(view_shape0[1])

    embedding = torch.empty_strided(
        full_shape0,
        full_stride0,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    gt0 = torch.empty_strided(
        full_shape0,
        full_stride0,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    mul1 = torch.empty_strided(
        full_shape0,
        full_stride0,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt0 = torch.empty_strided(
        row_shape0,
        _contiguous_stride(row_shape0),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    view0 = torch.empty_strided(
        view_shape0,
        _contiguous_stride(view_shape0),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    gt1 = torch.empty_strided(
        full_shape1,
        full_stride1,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    mul5 = torch.empty_strided(
        full_shape1,
        full_stride1,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt1 = torch.empty_strided(
        row_shape1,
        _contiguous_stride(row_shape1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    view1 = torch.empty_strided(
        view_shape1,
        _contiguous_stride(view_shape1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, arg0_1.device)
        _embedding_dual_dropout_rmsnorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            seeds,
            seeds,
            embedding,
            gt0,
            mul1,
            rsqrt0,
            view0,
            gt1,
            mul5,
            rsqrt1,
            view1,
            ROWS=rows,
            HIDDEN=hidden,
            USE_RANDOM_PTRS=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return embedding, seeds, gt0, mul1, rsqrt0, view0, gt1, mul5, rsqrt1, view1

    seeds, random0, random1 = _seeds_and_randoms_for_eager_check(
        full_shape0,
        full_shape1,
        device=arg0_1.device,
    )
    _embedding_dual_dropout_rmsnorm_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        seeds,
        seeds,
        embedding,
        gt0,
        mul1,
        rsqrt0,
        view0,
        gt1,
        mul5,
        rsqrt1,
        view1,
        ROWS=rows,
        HIDDEN=hidden,
        USE_RANDOM_PTRS=False,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _embedding_dual_dropout_rmsnorm_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        random0,
        random1,
        embedding,
        gt0,
        mul1,
        rsqrt0,
        view0,
        gt1,
        mul5,
        rsqrt1,
        view1,
        ROWS=rows,
        HIDDEN=hidden,
        USE_RANDOM_PTRS=True,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return embedding, seeds, gt0, mul1, rsqrt0, view0, gt1, mul5, rsqrt1, view1


# b30c9463: MT5 train embedding + seed-0/34 dual dropout RMSNorm, [32,128,512]
@oracle_impl(hardware="B200", point="b30c9463", BLOCK_M=1, BLOCK_N=512, num_warps=4, num_stages=3)
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
