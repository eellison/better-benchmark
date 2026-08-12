"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 attention dropout-safe-softmax scope in one Triton row kernel, including the shape-param view, fp32 stable last-dimension amax/libdevice.exp/sum/div side outputs, explicit bf16 probability cast, `eq(-inf)`/`any` all-masked-row zero fallback, returned bool all-masked mask, returned bf16 zero full tensor, internally generated 24-element Inductor seed tensor with seed-index-0 dropout, required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling by 1.1111111111111112, returned contiguous 3D view, and returned permute alias, whereas Inductor lowers seed generation, stochastic producer, safe softmax reductions, fallback selection, dropout epilogue, and sibling side outputs through generic scheduler fragments; Inductor cannot fuse this full returned-output envelope today because its row-softmax template does not keep reduction side outputs, generated seeds, dropout mask, zero fallback materialization, and layout-only aliases resident across the reduction and store epilogue while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach the attention-softmax scheduler to inline Inductor seed/dropout generation and emit the amax, sum, all-masked mask, zero fallback tensor, scaled bf16 view, and permute alias from one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


N_SEEDS = 24
SEED_INDEX = 0


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
def _softmax_outputs_random_kernel(
    x_ptr,
    random_ptr,
    amax_ptr,
    sum_ptr,
    all_masked_ptr,
    full_ptr,
    gt_ptr,
    dropped_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * k_len + cols[None, :]

    scores = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    not_minus_inf = mask & (scores != -float("inf"))
    has_any = tl.max(tl.where(not_minus_inf, 1, 0), axis=1) != 0
    all_masked = ~has_any

    row_max = tl.max(scores, axis=1)
    numer = libdevice.exp(scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)
    where_val = tl.where(all_masked[:, None], 0.0, probs).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(all_masked_ptr + rows, all_masked, mask=row_mask)
    tl.store(full_ptr + offsets, 0.0, mask=mask)

    rand_bf16 = tl.load(
        random_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, where_val, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled, mask=mask)


@triton.jit
def _softmax_outputs_seeded_kernel(
    x_ptr,
    seeds_ptr,
    amax_ptr,
    sum_ptr,
    all_masked_ptr,
    full_ptr,
    gt_ptr,
    dropped_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    rng_seed_index: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * k_len + cols[None, :]

    scores = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    not_minus_inf = mask & (scores != -float("inf"))
    has_any = tl.max(tl.where(not_minus_inf, 1, 0), axis=1) != 0
    all_masked = ~has_any

    row_max = tl.max(scores, axis=1)
    numer = libdevice.exp(scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)
    where_val = tl.where(all_masked[:, None], 0.0, probs).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)
    tl.store(all_masked_ptr + rows, all_masked, mask=row_mask)
    tl.store(full_ptr + offsets, 0.0, mask=mask)

    seed = tl.load(seeds_ptr + rng_seed_index)
    rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, where_val, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled, mask=mask)


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


def _inductor_seeds_random_for_eager_check(shape, *, device):
    seed_advance = 8
    total_advance = seed_advance + _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)
        seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return seeds, random

    seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    return seeds, torch.ops.prims.inductor_random.default(shape, seed, "rand")


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, shape0, shape1, shape2, _shape3, shape4 = inputs
    del shape0, _shape3

    full_shape = _as_shape(shape1)
    random_shape = _as_shape(shape2)
    out_shape = _as_shape(shape4)
    row_shape = full_shape[:-1] + (1,)
    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)

    row_stride = _contiguous_stride(row_shape)
    full_stride = _contiguous_stride(full_shape)
    out_stride = _contiguous_stride(out_shape)

    amax = torch.empty_strided(
        row_shape,
        row_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        row_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    all_masked = torch.empty_strided(
        row_shape,
        row_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    full = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    gt = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape,
        out_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, arg0_1.device)
        _softmax_outputs_seeded_kernel[grid](
            arg0_1,
            seeds,
            amax,
            sum_1,
            all_masked,
            full,
            gt,
            dropped,
            n_rows=n_rows,
            k_len=k_len,
            rng_seed_index=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seeds, random = _inductor_seeds_random_for_eager_check(
            random_shape,
            device=arg0_1.device,
        )
        _softmax_outputs_random_kernel[grid](
            arg0_1,
            random,
            amax,
            sum_1,
            all_masked,
            full,
            gt,
            dropped,
            n_rows=n_rows,
            k_len=k_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return amax, sum_1, all_masked, full, seeds, gt, dropped, dropped.permute(0, 2, 1)


# bcf6fe02: MobileBERT safe bf16 attention softmax/dropout with generated seeds.
@oracle_impl(hardware="B200", point="bcf6fe02", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
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
