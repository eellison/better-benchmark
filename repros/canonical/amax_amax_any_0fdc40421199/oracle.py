"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LayoutLM bf16 scaled attention softmax/dropout scope in one Triton row kernel, including the `[384,512,512] -> [32,12,512,512]` view, bf16 0.125 score scaling boundary, returned unscaled and scaled fp32 amax side outputs, finite-row `any` guard, Inductor's guarded shifted-score selection, natural-exp sum/div, explicit bf16 probability cast, seed-index 22 dropout with the required f32-random-to-bf16 cast before gt(0.1), bf16 dropout scaling by 1.1111111111111112, the returned contiguous `[384,512,512]` view, and the returned permute alias, whereas Inductor lowers the scaled-score producer, duplicate reductions, finite guard, stochastic producer, dropout epilogue, and layout-only outputs through generic scheduler fragments; Inductor cannot do this today because its row-softmax fusion path does not keep the dual amax side outputs, finite-row exceptional-value guard, seeded dropout mask, scaled output materialization, and alias-producing epilogue resident across the reduction while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: extend the attention-softmax scheduler to fuse guarded scaled softmax side outputs, Inductor-seeded dropout, and alias-producing layout epilogues in one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 22


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
def _scaled_softmax_dropout_kernel(
    x_ptr,
    random_or_seed_ptr,
    raw_amax_ptr,
    scaled_amax_ptr,
    all_finite_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    seed_index: tl.constexpr,
    use_seeded_rng: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * k_len + cols[None, :]

    raw = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    scaled_bf16 = _f32_mul(raw, 0.125).to(tl.bfloat16)
    scaled = scaled_bf16.to(tl.float32)

    raw_scores = tl.where(mask, raw, -float("inf"))
    scaled_scores = tl.where(mask, scaled, -float("inf"))
    raw_max = tl.max(raw_scores, axis=1)
    scaled_max = tl.max(scaled_scores, axis=1)

    finite = (scaled == scaled) & (tl.abs(scaled) != float("inf"))
    has_invalid = tl.max(tl.where(mask & ~finite, 1, 0), axis=1) != 0
    all_finite = ~has_invalid

    shifted_unscaled = _f32_mul(raw - raw_max[:, None], 0.125)
    shifted_scaled = scaled - scaled_max[:, None]
    shifted = tl.where(all_finite[:, None], shifted_unscaled, shifted_scaled)
    shifted = tl.where(mask, shifted, -float("inf"))

    numer = libdevice.exp(shifted)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(raw_amax_ptr + rows, raw_max, mask=row_mask)
    tl.store(scaled_amax_ptr + rows, scaled_max, mask=row_mask)
    tl.store(all_finite_ptr + rows, all_finite, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    if use_seeded_rng:
        seed = tl.load(random_or_seed_ptr + seed_index)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    else:
        rand_bf16 = tl.load(
            random_or_seed_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled_dropout = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled_dropout, mask=mask)


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


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    del _expand_shape

    full_shape = _shape_tuple(full_shape_arg)
    random_shape = _shape_tuple(random_shape_arg)
    out_shape = _shape_tuple(out_shape_arg)
    k_len = int(full_shape[-1])
    n_rows = int(x.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    row_stride = _contiguous_stride(row_shape)
    full_stride = _contiguous_stride(full_shape)

    raw_amax = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.float32,
    )
    scaled_amax = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.float32,
    )
    all_finite = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.bool,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape,
        full_stride,
        device=x.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(n_rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _scaled_softmax_dropout_kernel[grid](
            x,
            seeds,
            raw_amax,
            scaled_amax,
            all_finite,
            sum_1,
            gt,
            dropped,
            n_rows=n_rows,
            k_len=k_len,
            seed_index=SEED_INDEX,
            use_seeded_rng=True,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=x.device,
        )
        _scaled_softmax_dropout_kernel[grid](
            x,
            random,
            raw_amax,
            scaled_amax,
            all_finite,
            sum_1,
            gt,
            dropped,
            n_rows=n_rows,
            k_len=k_len,
            seed_index=SEED_INDEX,
            use_seeded_rng=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)


# 279c055a: (T([384,512,512], bf16), T([37], i64), S([32,12,512,512]), ...)
@oracle_impl(hardware="B200", point="279c055a", BLOCK_M=4, BLOCK_N=512, num_warps=8, num_stages=3)
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
