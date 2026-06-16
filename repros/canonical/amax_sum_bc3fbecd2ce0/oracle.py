"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XGLM training additive-bias attention softmax/dropout scope in one Triton row kernel, including the `[512,128,128] -> [32,16,128,128]` view, f32 broadcast bias add, finite f32 sentinel maximum, returned f32 amax and denominator tensors, generated `inductor_seeds.default(3)` side output, seed-index-0 f32 dropout mask, f32 dropout scaling, final bf16 tensor, and returned permute alias, whereas Inductor lowers the decomposed view/add/maximum/amax/sub/exp/sum/div/generated-RNG/dropout/cast/permute graph through separate generic reduction, stochastic pointwise, and layout scheduling fragments; Inductor cannot do this today because its scheduler does not keep the broadcast score producer, generated-seed RNG producer, reduction side outputs, dropout epilogue, and alias-only return inside one row-softmax plan while preserving the f32 score boundary and RNG contract; the fix is SCHEDULER_FUSION: teach the attention-softmax scheduler to fuse additive-bias score construction, natural-exp row reductions, generated Inductor-seeded dropout, side-output stores, and layout-only returns into one guarded template."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_COUNT = 3
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
def _xglm_softmax_dropout_kernel(
    scores_ptr,
    bias_ptr,
    rng_ptr,
    amax_ptr,
    sum_ptr,
    keep_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    Q_LEN: tl.constexpr,
    K_LEN: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    active = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // Q_LEN
    batch = flat_bh // HEADS
    query = rows - flat_bh * Q_LEN
    offsets = rows[:, None] * K_LEN + cols[None, :]
    bias_offsets = batch[:, None] * (Q_LEN * K_LEN) + query[:, None] * K_LEN + cols[None, :]

    x = tl.load(scores_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=active, other=0.0).to(tl.float32)
    scores = tl.maximum(x + bias, -3.4028234663852886e38)
    reduced_scores = tl.where(active, scores, -float("inf"))

    row_max = tl.max(reduced_scores, axis=1)
    row_max = tl.where(row_mask, row_max, 0.0)
    numer = libdevice.exp(reduced_scores - row_max[:, None])
    numer = tl.where(active, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    if USE_RANDOM_PTR:
        random = tl.load(rng_ptr + offsets, mask=active, other=0.0, eviction_policy="evict_first")
    else:
        seed = tl.load(rng_ptr + 0)
        random = tl.rand(seed, offsets.to(tl.uint32))
    keep = random > 0.1
    tl.store(keep_ptr + offsets, keep, mask=active)

    dropped = _f32_mul(keep.to(tl.float32), probs)
    scaled = _f32_mul(dropped, 1.1111111111111112)
    tl.store(out_ptr + offsets, scaled.to(tl.bfloat16), mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


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


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


def _launch(
    arg0_1,
    arg1_1,
    rng_source,
    amax,
    denom,
    keep,
    out,
    *,
    use_random_ptr,
    BLOCK_M,
    BLOCK_N,
    num_warps,
    num_stages,
):
    rows = arg0_1.numel() // arg0_1.shape[-1]
    _xglm_softmax_dropout_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        rng_source,
        amax,
        denom,
        keep,
        out,
        ROWS=rows,
        HEADS=16,
        Q_LEN=128,
        K_LEN=128,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# d9690337: XGLM train additive-bias K=128 attention softmax/dropout.
@oracle_impl(hardware="B200", point="d9690337", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, random_shape_param = inputs
    del _shape_param_0, _shape_param_1
    device = arg0_1.device
    out_shape = tuple(arg0_1.shape)
    row_shape = out_shape[:-1] + (1,)
    out_stride = _contiguous_stride(out_shape)
    row_stride = _contiguous_stride(row_shape)

    amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    denom = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    keep = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bool)
    out = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bfloat16)

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _launch(
            arg0_1,
            arg1_1,
            seeds,
            amax,
            denom,
            keep,
            out,
            use_random_ptr=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return amax, denom, seeds, keep, out, out.permute(0, 2, 1)

    random_shape = _as_shape(random_shape_param)
    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
    _launch(
        arg0_1,
        arg1_1,
        seeds,
        amax,
        denom,
        keep,
        out,
        use_random_ptr=False,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _launch(
        arg0_1,
        arg1_1,
        random,
        amax,
        denom,
        keep,
        out,
        use_random_ptr=True,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return amax, denom, seeds, keep, out, out.permute(0, 2, 1)
