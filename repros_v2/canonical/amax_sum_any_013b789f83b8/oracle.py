"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 BERT/Electra attention softmax/dropout scope, including the `[B*H,512,512]` to `[B,H,512,512]` score view, stable fp32 natural-exp softmax, all-minus-inf row fallback to the bf16 input tensor, explicit bf16 probability rounding, Inductor seed-index-25 bf16 RNG mask, dropout scaling, contiguous flattened view output, and returned permute alias in one Triton row kernel, whereas Inductor lowers the decomposed view/cast/amax/sub/exp/sum/div/cast/eq/any/where/inductor_random/dropout/view/permute graph through generic reduction, RNG, pointwise, and layout scheduling; Inductor cannot do this today because its pattern library does not recognize a stochastic bf16 attention-softmax template with observable pre-dropout probabilities, observable bool mask, all-minus-inf fallback, and sibling layout-only outputs as one full-scope row plan; the fix is NEW_PATTERN: add a guarded bf16 attention-softmax-dropout lowering that emits the pre-dropout bf16 tensor, exact seeded mask, flattened dropout storage, and metadata permute alias from one fused schedule."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 25


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
    scores_ptr,
    fallback_ptr,
    rng_ptr,
    where_ptr,
    gt_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    K_LEN: tl.constexpr,
    RNG_SEED_INDEX: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < ROWS
    col_mask = cols < K_LEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * K_LEN + cols[None, :]

    raw_scores = tl.load(scores_ptr + offsets, mask=mask, other=-float("inf"))
    scores = raw_scores.to(tl.float32)
    present = tl.where(mask & (scores != -float("inf")), 1, 0)
    row_has_value = tl.sum(present, axis=1) > 0

    row_max = tl.max(tl.where(mask, scores, -float("inf")), axis=1)
    safe_max = tl.where(row_has_value, row_max, 0.0)
    numer = libdevice.exp(scores - safe_max[:, None])
    numer = tl.where(mask & row_has_value[:, None], numer, 0.0)
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(row_has_value, denom, 1.0)
    probs_bf16 = (numer / safe_denom[:, None]).to(tl.bfloat16)

    fallback = tl.load(
        fallback_ptr + offsets,
        mask=mask & (row_has_value[:, None] == 0),
        other=0.0,
    )
    where_value = tl.where(row_has_value[:, None], probs_bf16, fallback)
    tl.store(where_ptr + offsets, where_value, mask=mask)

    if USE_RANDOM_PTR:
        rand_bf16 = tl.load(rng_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
    else:
        seed = tl.load(rng_ptr + RNG_SEED_INDEX)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_K), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, where_value, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(out_ptr + offsets, scaled, mask=mask)


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


def _torch_rand_uniform_offset_advance(numel, *, device):
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        props.multi_processor_count * blocks_per_sm,
        (int(numel) + block_size - 1) // block_size,
    )
    unroll = 4
    counter_offset = (
        (int(numel) - 1) // (block_size * grid * unroll) + 1
    ) * 4
    return counter_offset * 2


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = _torch_rand_uniform_offset_advance(numel, device=device)
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


def _launch_softmax_dropout(
    scores,
    fallback,
    rng,
    where,
    gt,
    out,
    *,
    use_random_ptr,
    block_m,
    block_k,
    num_warps,
    num_stages,
):
    rows = int(scores.numel() // scores.shape[-1])
    k_len = int(scores.shape[-1])
    _softmax_dropout_kernel[(triton.cdiv(rows, block_m),)](
        scores,
        fallback,
        rng,
        where,
        gt,
        out,
        ROWS=rows,
        K_LEN=k_len,
        RNG_SEED_INDEX=SEED_INDEX,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# 8ae0f618: (T([384,512,512], bf16), T([32,12,512,512], bf16), T([37], i64), ...)
@oracle_impl(hardware="B200", point="8ae0f618", block_m=4, block_k=512, num_warps=8, num_stages=3)
# fac7e171: (T([256,512,512], bf16), T([64,4,512,512], bf16), T([37], i64), ...)
@oracle_impl(hardware="B200", point="fac7e171", block_m=4, block_k=512, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    view_shape = _as_shape(_shape_param_0)
    flat_shape = _as_shape(_shape_param_3)

    where = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    gt = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        _launch_softmax_dropout(
            arg0_1,
            arg1_1,
            arg2_1,
            where,
            gt,
            out,
            use_random_ptr=False,
            block_m=block_m,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return where, gt, out, out.permute(0, 2, 1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        _as_shape(_shape_param_1),
        seed,
        device=arg0_1.device,
    )
    _launch_softmax_dropout(
        arg0_1,
        arg1_1,
        random,
        where,
        gt,
        out,
        use_random_ptr=True,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return where, gt, out, out.permute(0, 2, 1)
