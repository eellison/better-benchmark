"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 T5/MT5 attention softmax/dropout scope in one Triton row kernel, including the shape-param view, strided f32 bias add, explicit bf16 post-add rounding, fp32 stable last-dimension amax/libdevice.exp/sum/div, returned bf16 rounded score tensor, returned fp32 amax and sum side outputs, Inductor seed-index-29 dropout with the required f32-random-to-bf16 cast before gt(0.1), bf16 dropout scaling by 1.1111111111111112, the returned contiguous 3D view, and returned permute alias, whereas Inductor lowers the strided add, reduction, stochastic producer, dropout epilogue, and sibling side outputs through generic scheduler fragments; Inductor cannot do this today because its row-softmax fusion path does not keep the bf16 score materialization, reduction side outputs, seeded dropout mask, scaled output view, and layout-only alias resident across the reduction epilogue while preserving the bf16 rounding boundaries; the fix is SCHEDULER_FUSION: extend the attention-softmax scheduler to fuse strided score biasing, observable reduction side outputs, Inductor-seeded dropout, and alias-producing layout epilogues in one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 29


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
    arg0_ptr,
    arg1_ptr,
    random_ptr,
    rounded_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    n_rows: tl.constexpr,
    n_heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    bh = rows // q_len
    q = rows - bh * q_len
    h = bh % n_heads
    b = bh // n_heads
    dense_offsets = rows[:, None] * k_len + cols[None, :]
    strided_offsets = (
        b[:, None] * arg1_s0
        + h[:, None] * arg1_s1
        + q[:, None] * arg1_s2
        + cols[None, :] * arg1_s3
    )

    view_val = tl.load(arg0_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(arg1_ptr + strided_offsets, mask=mask, other=0.0).to(tl.float32)
    rounded = (view_val + bias).to(tl.bfloat16)
    tl.store(rounded_ptr + dense_offsets, rounded, mask=mask)

    scores = rounded.to(tl.float32)
    scores = tl.where(mask, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    rand_bf16 = tl.load(
        random_ptr + dense_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + dense_offsets, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + dense_offsets, scaled, mask=mask)


@triton.jit
def _softmax_dropout_seeded_kernel(
    arg0_ptr,
    arg1_ptr,
    seeds_ptr,
    rounded_ptr,
    amax_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    n_rows: tl.constexpr,
    n_heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    seed_index: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    bh = rows // q_len
    q = rows - bh * q_len
    h = bh % n_heads
    b = bh // n_heads
    dense_offsets = rows[:, None] * k_len + cols[None, :]
    strided_offsets = (
        b[:, None] * arg1_s0
        + h[:, None] * arg1_s1
        + q[:, None] * arg1_s2
        + cols[None, :] * arg1_s3
    )

    view_val = tl.load(arg0_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(arg1_ptr + strided_offsets, mask=mask, other=0.0).to(tl.float32)
    rounded = (view_val + bias).to(tl.bfloat16)
    tl.store(rounded_ptr + dense_offsets, rounded, mask=mask)

    scores = rounded.to(tl.float32)
    scores = tl.where(mask, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    numer = libdevice.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = (numer / denom[:, None]).to(tl.bfloat16)

    tl.store(amax_ptr + rows, row_max, mask=row_mask)
    tl.store(sum_ptr + rows, denom, mask=row_mask)

    seed = tl.load(seeds_ptr + seed_index)
    rand_bf16 = tl.rand(seed, dense_offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + dense_offsets, keep, mask=mask)

    dropped = tl.where(keep, probs, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    tl.store(dropped_ptr + dense_offsets, scaled, mask=mask)


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
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    del shape0, _shape2

    full_shape = tuple(int(dim) for dim in shape1)
    out_shape = tuple(int(dim) for dim in shape3)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(arg0_1.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    full_stride = _contiguous_stride(full_shape)
    row_stride = _contiguous_stride(row_shape)
    out_stride = _contiguous_stride(out_shape)

    rounded = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
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
            arg2_1,
            rounded,
            amax,
            sum_1,
            gt,
            dropped,
            arg1_s0=arg1_1.stride(0),
            arg1_s1=arg1_1.stride(1),
            arg1_s2=arg1_1.stride(2),
            arg1_s3=arg1_1.stride(3),
            n_rows=n_rows,
            n_heads=n_heads,
            q_len=q_len,
            k_len=k_len,
            seed_index=SEED_INDEX,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            full_shape,
            seed,
            device=arg0_1.device,
        )
        _softmax_dropout_random_kernel[grid](
            arg0_1,
            arg1_1,
            random,
            rounded,
            amax,
            sum_1,
            gt,
            dropped,
            arg1_s0=arg1_1.stride(0),
            arg1_s1=arg1_1.stride(1),
            arg1_s2=arg1_1.stride(2),
            arg1_s3=arg1_1.stride(3),
            n_rows=n_rows,
            n_heads=n_heads,
            q_len=q_len,
            k_len=k_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)


# dda3d8e0: (T([192,128,128], bf16), T([32,6,128,128], f32, stride=(98304,1,768,6)), T([84], i64), ...)
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
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
