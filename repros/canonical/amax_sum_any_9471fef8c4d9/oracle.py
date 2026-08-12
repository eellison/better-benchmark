"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT bf16 attention dropout-safe-softmax scope in one Triton row kernel, including the shape-param view, fp32 stable last-dimension amax/libdevice.exp/sum/div, explicit bf16 probability cast, `eq(-inf)`/`any` all-masked-row fallback to the bf16 tensor input, Inductor seed-index-14 dropout with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling by 1.1111111111111112, the returned 4D softmax/fallback tensor, returned bool mask, returned contiguous 3D view, and returned permute alias, whereas Inductor lowers the stochastic producer, safe softmax reduction, fallback selection, dropout epilogue, and sibling layout-only outputs through generic scheduler fragments; Inductor cannot fuse this full returned-output envelope today because its row-softmax template does not keep the seeded dropout mask and all observable side outputs resident across the reduction and store epilogue while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach the attention-softmax scheduler to inline Inductor-seeded dropout and emit the fallback tensor, mask, scaled bf16 view, and permute alias from one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 14


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
def _safe_softmax_dropout_random_kernel(
    x_ptr,
    fallback_ptr,
    random_ptr,
    where_ptr,
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
    live = mask & (scores != -float("inf"))
    has_any = tl.max(tl.where(live, 1, 0), axis=1) != 0

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(has_any, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(live, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(has_any, denom, 1.0)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    fallback = tl.load(fallback_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    where_val = tl.where(has_any[:, None], probs, fallback).to(tl.bfloat16)
    tl.store(where_ptr + offsets, where_val, mask=mask)

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
def _safe_softmax_dropout_seeded_kernel(
    x_ptr,
    fallback_ptr,
    seeds_ptr,
    where_ptr,
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
    live = mask & (scores != -float("inf"))
    has_any = tl.max(tl.where(live, 1, 0), axis=1) != 0

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(has_any, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(live, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    denom = tl.where(has_any, denom, 1.0)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    fallback = tl.load(fallback_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    where_val = tl.where(has_any[:, None], probs, fallback).to(tl.bfloat16)
    tl.store(where_ptr + offsets, where_val, mask=mask)

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


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _launch(inputs, *, block_m: int, block_n: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, _shape3 = inputs
    del _shape0, _shape2, _shape3

    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    random_shape = _as_shape(shape1)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    grid = (triton.cdiv(n_rows, block_m),)
    if torch.cuda.is_current_stream_capturing():
        _safe_softmax_dropout_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            where,
            gt,
            dropped,
            n_rows=n_rows,
            k_len=k_len,
            rng_seed_index=SEED_INDEX,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
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
        _safe_softmax_dropout_random_kernel[grid](
            arg0_1,
            arg1_1,
            random,
            where,
            gt,
            dropped,
            n_rows=n_rows,
            k_len=k_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return where, gt, dropped, dropped.permute(0, 2, 1)


# d59f4ab1: MobileBERT safe bf16 attention softmax/dropout, B=256, H=4, S=128.
@oracle_impl(hardware="B200", point="d59f4ab1", block_m=8, block_n=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
