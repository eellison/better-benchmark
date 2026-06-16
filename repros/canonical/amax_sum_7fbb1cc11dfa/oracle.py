"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MT5 attention softmax/dropout scope in one Triton row kernel, including the `[192,128,128] -> [32,6,128,128]` view, bf16-to-fp32 promotion, stable last-dimension amax/libdevice.exp/sum/div, returned fp32 amax and sum side outputs, explicit bf16 probability cast, Inductor seed-index 67 dropout with the required f32-random-to-bf16 cast before gt(0.1), bf16 dropout scaling by 1.1111111111111112, the returned contiguous `[192,128,128]` view, and the returned permute alias, whereas Inductor lowers the decomposed reduction, stochastic producer, dropout epilogue, and layout-only outputs through generic scheduler fragments; Inductor cannot do this today because its row-softmax fusion path does not keep observable reduction side outputs, seeded dropout mask generation, scaled output materialization, and alias-producing layout epilogues resident across the reduction while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: extend the attention-softmax scheduler to fuse observable reduction side outputs, Inductor-seeded dropout, and alias-producing layout epilogues in one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 67


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
def _softmax_dropout_kernel(
    x_ptr,
    random_or_seed_ptr,
    amax_ptr,
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

    scores = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    numer = libdevice.exp(scores)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
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
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled, mask=mask)


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

    amax = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=x.device,
        dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=x.device,
        dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
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
        _softmax_dropout_kernel[grid](
            x,
            seeds,
            amax,
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
        _softmax_dropout_kernel[grid](
            x,
            random,
            amax,
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

    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)


# 1715052e: (T([192,128,128], bf16), T([84], i64), S([32,6,128,128]), ...)
@oracle_impl(hardware="B200", point="1715052e", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
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
