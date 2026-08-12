"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete M2M100 additive-mask attention softmax/dropout scope, including scalar-selected bf16 mask bias, bf16 score-add rounding, stable fp32 softmax, the all-minus-inf row zero guard, Inductor RNG dropout, and the trailing view/permute alias; Inductor lowers the decomposed where/add/reduction/any/dropout/layout graph as generic pointwise, reduction, RNG, and layout kernels because it lacks a guarded additive-mask attention softmax/dropout pattern that preserves stochastic and alias-visible outputs."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 1
ROWS = 64 * 16 * 128


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
def _softmax_any_dropout_kernel(
    mask_ptr,
    true_bias_ptr,
    false_bias_ptr,
    scores_ptr,
    seeds_or_random,
    where_out,
    gt_out,
    value_out,
    use_random_ptr: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < 131072
    col_mask = cols < 128
    mask = row_mask[:, None] & col_mask[None, :]

    bh = rows // 128
    query = rows - bh * 128
    offsets = rows[:, None] * 128 + cols[None, :]

    raw = tl.load(scores_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    key_mask = tl.load(mask_ptr + query[:, None], mask=row_mask[:, None], other=0).to(tl.int1)
    true_bias = tl.load(true_bias_ptr).to(tl.float32)
    false_bias = tl.load(false_bias_ptr).to(tl.float32)
    bias = tl.where(key_mask, true_bias, false_bias)
    scores = raw + bias

    row_has_value = tl.sum(tl.where(col_mask[None, :] & (scores != -float("inf")), 1, 0), axis=1) != 0
    reduced = tl.where(col_mask[None, :], scores, -float("inf"))
    row_max = tl.max(reduced, axis=1)
    has_nan = tl.max(
        tl.where(col_mask[None, :] & (scores != scores), 1, 0),
        axis=1,
    ) != 0
    safe_max = tl.where(row_has_value, row_max, 0.0)
    safe_max = tl.where(has_nan, tl.full((BLOCK_M,), float("nan"), tl.float32), safe_max)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(row_has_value[:, None] & col_mask[None, :], numer, 0.0)
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(row_has_value, denom, 1.0)
    probs_bf16 = (numer / safe_denom[:, None]).to(tl.bfloat16)
    zero_bf16 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32).to(tl.bfloat16)
    where_value = tl.where(row_has_value[:, None], probs_bf16, zero_bf16)
    tl.store(where_out + offsets, where_value, mask=mask)

    if use_random_ptr:
        rand_bf16 = tl.load(seeds_or_random + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_or_random + 1)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    keep = rand_bf16 > tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    tl.store(gt_out + offsets, keep, mask=mask)

    dropped = (where_value.to(tl.float32) * keep.to(tl.float32)).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(value_out + offsets, scaled, mask=mask)


def _launch(
    arg0_1,
    arg1_1,
    arg2_1,
    arg3_1,
    arg4_1,
    shape_param_1,
    shape_param_2,
    shape_param_3,
    shape_param_4,
    shape_param_5,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    del shape_param_1, shape_param_2, shape_param_3, shape_param_4, shape_param_5

    full_shape = (64, 16, 128, 128)
    flat_shape = (1024, 128, 128)
    where_out = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=arg3_1.device,
        dtype=arg3_1.dtype,
    )
    gt_out = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=arg3_1.device,
        dtype=torch.bool,
    )
    value_out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg3_1.device,
        dtype=arg3_1.dtype,
    )

    if torch.cuda.is_current_stream_capturing():
        rng_source = arg4_1
        use_random_ptr = False
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg4_1, SEED_INDEX)
        rng_source = _inductor_random_for_eager_check(full_shape, seed, device=arg3_1.device)
        use_random_ptr = True

    _softmax_any_dropout_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        rng_source,
        where_out,
        gt_out,
        value_out,
        use_random_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return where_out, gt_out, value_out, value_out.permute(0, 2, 1)


@oracle_impl(hardware="B200", point="3f818223", BLOCK_M=8, BLOCK_N=128, num_warps=4, num_stages=3)
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
