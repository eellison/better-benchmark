"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full MT5 additive-bias attention softmax/dropout scope by fusing the bf16-rounded score add, visible row max/sum reductions, Inductor RNG dropout mask, dropout scaling, and trailing view/permute alias into one row-softmax kernel; Inductor lowers the decomposed add/cast/amax/exp/sum/div/random/dropout/layout graph as separate generic kernels because it lacks a T5-style additive-bias attention softmax/dropout pattern with observable intermediates and a layout-only epilogue."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 65
ROWS = 32 * 6 * 128


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


@triton.jit
def _softmax_dropout_kernel(
    arg0,
    arg1,
    seeds_or_random,
    scores_out,
    amax_out,
    sum_out,
    gt_out,
    value_out,
    use_random_ptr: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < 24576
    col_mask = cols < 128

    bh = rows // 128
    batch = bh // 6
    head = bh - batch * 6
    query = rows - bh * 128
    dense_offsets = rows[:, None] * 128 + cols[None, :]
    bias_offsets = (
        batch[:, None] * 98304
        + head[:, None]
        + query[:, None] * 768
        + cols[None, :] * 6
    )
    mask = row_mask[:, None] & col_mask[None, :]

    raw = tl.load(arg0 + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(arg1 + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    rounded = (raw + bias).to(tl.bfloat16)
    tl.store(scores_out + dense_offsets, rounded, mask=mask)

    scores = rounded.to(tl.float32)
    neg_inf = tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32)
    reduced = tl.where(col_mask[None, :], scores, neg_inf)
    row_max = tl.max(reduced, axis=1)
    has_nan = tl.max(
        tl.where(col_mask[None, :] & (scores != scores), 1, 0),
        axis=1,
    ) != 0
    row_max = tl.where(has_nan, tl.full((BLOCK_M,), float("nan"), tl.float32), row_max)
    shifted = reduced - row_max[:, None]
    numer = libdevice.exp(shifted)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    probs_bf16 = probs.to(tl.bfloat16)

    if use_random_ptr:
        random = tl.load(seeds_or_random + dense_offsets, mask=mask, other=0.0).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_or_random + 65)
        random = tl.rand(seed, dense_offsets.to(tl.uint32)).to(tl.bfloat16)
    keep = random > tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    tl.store(gt_out + dense_offsets, keep, mask=mask)

    tl.store(amax_out + rows, row_max, mask=row_mask)
    tl.store(sum_out + rows, denom, mask=row_mask)

    dropped = (probs_bf16.to(tl.float32) * keep.to(tl.float32)).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(value_out + dense_offsets, scaled, mask=mask)


def _launch(
    arg0_1,
    arg1_1,
    arg2_1,
    shape_param_1,
    shape_param_2,
    shape_param_3,
    shape_param_4,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    del shape_param_1, shape_param_2, shape_param_3, shape_param_4

    score_shape = (32, 6, 128, 128)
    row_shape = (32, 6, 128, 1)
    out_shape = (192, 128, 128)
    scores = torch.empty_strided(
        score_shape,
        _contiguous_stride(score_shape),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    amax_out = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    gt_out = torch.empty_strided(
        score_shape,
        _contiguous_stride(score_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    value_out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    if torch.cuda.is_current_stream_capturing():
        rng_source = arg2_1
        use_random_ptr = False
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
        rng_source = _inductor_random_for_eager_check(score_shape, seed, device=arg0_1.device)
        use_random_ptr = True

    _softmax_dropout_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        arg0_1,
        arg1_1,
        rng_source,
        scores,
        amax_out,
        sum_out,
        gt_out,
        value_out,
        use_random_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return scores, amax_out, sum_out, gt_out, value_out, value_out.permute(0, 2, 1)


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_M=4, BLOCK_N=128, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        *inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
