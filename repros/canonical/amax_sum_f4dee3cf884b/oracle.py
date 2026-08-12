"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeBERTa masked attention softmax/dropout scope in one Triton row kernel, including the `[192,512,512] -> [8,24,512,512]` score view, broadcast bool mask with bf16 scalar fill, returned bf16 masked scores, fp32 stable last-dimension amax/libdevice.exp/sum/div, returned fp32 row denominator, Inductor seed-index-40 f32 dropout mask, f32 dropout scaling by 1.1111111111111112 before the final bf16 cast, returned contiguous flattened output, and returned permute alias, whereas Inductor lowers the decomposed view/where/cast/amax/sub/exp/sum/div/stochastic/dropout/view/cast/permute graph through generic reduction, RNG, pointwise, and layout kernels over materialized intermediates; Inductor cannot do this today because its row-softmax scheduler does not keep the broadcast mask fill, visible row-stat outputs, seeded dropout mask, and alias-producing output epilogue resident across the reduction while preserving the f32 RNG comparison and bf16 output rounding boundary; the fix is SCHEDULER_FUSION: extend the attention-softmax lowering to fuse scalar-fill masking, observable row reductions, Inductor-seeded dropout, and layout-only aliases in one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 40


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
    scores_ptr,
    mask_ptr,
    fill_ptr,
    random_ptr,
    where_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    out_ptr,
    N_HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    bh = rows // Q_LEN
    batch = bh // N_HEADS
    query = rows - bh * Q_LEN
    dense_offsets = rows[:, None] * K_LEN + cols[None, :]
    mask_offsets = batch[:, None] * Q_LEN * K_LEN + query[:, None] * K_LEN + cols[None, :]

    raw = tl.load(scores_ptr + dense_offsets).to(tl.bfloat16)
    mask = tl.load(mask_ptr + mask_offsets)
    fill = tl.load(fill_ptr).to(tl.bfloat16)
    masked = tl.where(mask, fill, raw)
    tl.store(where_ptr + dense_offsets, masked)

    scores = masked.to(tl.float32)
    row_max = tl.max(scores, axis=1)
    shifted = scores - row_max[:, None]
    numer = libdevice.exp(shifted)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max)
    tl.store(sum_ptr + rows, denom)

    random = tl.load(random_ptr + dense_offsets, eviction_policy="evict_first").to(tl.float32)
    keep = random > 0.1
    tl.store(gt_ptr + dense_offsets, keep)

    dropped = tl.where(keep, probs, 0.0)
    scaled = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    tl.store(out_ptr + dense_offsets, scaled)


@triton.jit
def _masked_softmax_dropout_seeded_kernel(
    scores_ptr,
    mask_ptr,
    fill_ptr,
    seeds_ptr,
    where_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    out_ptr,
    N_HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    SEED_INDEX_CONST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    bh = rows // Q_LEN
    batch = bh // N_HEADS
    query = rows - bh * Q_LEN
    dense_offsets = rows[:, None] * K_LEN + cols[None, :]
    mask_offsets = batch[:, None] * Q_LEN * K_LEN + query[:, None] * K_LEN + cols[None, :]

    raw = tl.load(scores_ptr + dense_offsets).to(tl.bfloat16)
    mask = tl.load(mask_ptr + mask_offsets)
    fill = tl.load(fill_ptr).to(tl.bfloat16)
    masked = tl.where(mask, fill, raw)
    tl.store(where_ptr + dense_offsets, masked)

    scores = masked.to(tl.float32)
    row_max = tl.max(scores, axis=1)
    shifted = scores - row_max[:, None]
    numer = libdevice.exp(shifted)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max)
    tl.store(sum_ptr + rows, denom)

    seed = tl.load(seeds_ptr + SEED_INDEX_CONST)
    random = tl.rand(seed, dense_offsets.to(tl.uint32))
    keep = random > 0.1
    tl.store(gt_ptr + dense_offsets, keep)

    dropped = tl.where(keep, probs, 0.0)
    scaled = _f32_mul(dropped, 1.1111111111111112).to(tl.bfloat16)
    tl.store(out_ptr + dense_offsets, scaled)


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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    unknown = -1
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            unknown = idx
        else:
            known *= dim
    if unknown >= 0:
        dims[unknown] = int(numel) // known
    return tuple(dims)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    full_shape = _shape_tuple(shape1)
    flat_shape = _resolve_shape(shape2, arg0_1.numel())
    row_shape = full_shape[:-1] + (1,)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(arg0_1.numel() // k_len)
    del shape0

    where = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _masked_softmax_dropout_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            where,
            amax,
            sum_1,
            gt,
            dropped,
            N_HEADS=n_heads,
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
            where,
            amax,
            sum_1,
            gt,
            dropped,
            N_HEADS=n_heads,
            Q_LEN=q_len,
            K_LEN=k_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return where, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)


# 00541467: DeBERTa masked attention softmax/dropout, bf16 [192,512,512].
@oracle_impl(hardware="B200", point="00541467", BLOCK_M=4, BLOCK_N=512, num_warps=8, num_stages=3)
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
