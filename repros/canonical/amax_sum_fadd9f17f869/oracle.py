"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 T5 attention softmax/dropout scope in one Triton row kernel, including the shape-param view, bf16-to-fp32 score promotion, stable last-dimension amax/libdevice.exp/sum/div, returned fp32 amax and sum side outputs, explicit bf16 probability cast, Inductor seed-index-53 dropout with the required f32-random-to-bf16 comparison, bf16 dropout scaling by 1.1111111111111112, the returned contiguous 3D view, and returned permute alias, whereas Inductor lowers the decomposed view/cast/amax/sub/exp/sum/div/cast/RNG/dropout/view/permute graph through separate generic reduction, stochastic pointwise, and layout scheduling fragments; Inductor cannot do this today because its scheduler does not keep the seeded dropout producer and all observable side outputs inside one row-softmax plan while preserving the bf16 rounding boundaries and layout-only aliases; the fix is SCHEDULER_FUSION: teach the attention-softmax scheduler to fuse bf16 row softmax, observable reduction side outputs, Inductor-seeded dropout, and alias-producing layout epilogues in one guarded row template."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 53


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
    x_ptr,
    random_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * k_len + cols[None, :]

    scores = tl.load(x_ptr + offsets).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    numer = libdevice.exp(scores)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max)
    tl.store(sum_ptr + rows, denom)

    rand_bf16 = tl.load(random_ptr + offsets, eviction_policy="evict_first").to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled)


@triton.jit
def _softmax_dropout_seeded_kernel(
    x_ptr,
    seeds_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    k_len: tl.constexpr,
    seed_index: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * k_len + cols[None, :]

    scores = tl.load(x_ptr + offsets).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    numer = libdevice.exp(scores)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max)
    tl.store(sum_ptr + rows, denom)

    seed = tl.load(seeds_ptr + seed_index)
    rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + offsets, scaled)


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


def _launch(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, shape0, shape1, _shape2, shape3 = inputs
    del _shape2

    full_shape = tuple(int(dim) for dim in shape0)
    random_shape = tuple(int(dim) for dim in shape1)
    out_shape = tuple(int(dim) for dim in shape3)
    row_shape = full_shape[:-1] + (1,)
    k_len = int(full_shape[-1])
    n_rows = int(arg0_1.numel() // k_len)

    full_stride = _contiguous_stride(full_shape)
    row_stride = _contiguous_stride(row_shape)
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
        _softmax_dropout_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            amax,
            sum_1,
            gt,
            dropped,
            k_len=k_len,
            seed_index=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=arg0_1.device,
        )
        _softmax_dropout_random_kernel[grid](
            arg0_1,
            random,
            amax,
            sum_1,
            gt,
            dropped,
            k_len=k_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)


# 696b5761: (T([64,1024,1024], bf16), T([64], i64), S([8,8,1024,1024]), ...)
@oracle_impl(hardware="B200", point="696b5761", BLOCK_M=1, BLOCK_N=1024, num_warps=2, num_stages=3)
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
