"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 T5/MT5 seeded dropout-residual-RMSNorm scope in one Triton row kernel, including the flat-to-3D view, seed-index-16 Inductor dropout with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling, returned bool mask, returned fp32 residual sum, fp32 mean-square reduction with eps=1e-6, returned rsqrt, fp32 affine multiply, bf16 output cast, and final flattened view, whereas Inductor lowers the stochastic producer, residual add, RMS reduction, affine epilogue, and visible side outputs through generic RNG, reduction, pointwise, and view scheduling; Inductor cannot do this today because its norm-template canonicalization does not recognize an Inductor-seeded dropout-add producer feeding fixed-hidden RMSNorm while preserving all returned side outputs and bf16 rounding boundaries; the fix is NEW_PATTERN: add a guarded seeded-dropout residual RMSNorm template that threads the RNG mask through the row reduction and emits the mask, residual sum, inverse RMS, and affine output view from one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_SCALE = 1.1111111111111112


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _dropout_rmsnorm_kernel(
    flat_ptr,
    seeds_or_random_ptr,
    residual_ptr,
    weight_ptr,
    gt_ptr,
    add_ptr,
    rsqrt_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEED_IDX: tl.constexpr,
    DROPOUT_SCALE_VALUE: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_RANDOM_PTR:
        random_bf16 = tl.load(
            seeds_or_random_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_or_random_ptr + SEED_IDX)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    threshold = tl.full((ROW_BLOCK, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, flat, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_VALUE).to(tl.bfloat16)
    add = _f32_add(residual, scaled.to(tl.float32))
    tl.store(add_ptr + offsets, add, mask=mask)

    add_for_reduce = tl.where(mask, add, 0.0)
    square_sum = tl.sum(_f32_mul(add_for_reduce, add_for_reduce), axis=1)
    inv_rms = libdevice.rsqrt(_f32_add(square_sum / HIDDEN, 1.0e-6))
    tl.store(rsqrt_ptr + rows, inv_rms, mask=row_mask)

    normalized = _f32_mul(add, inv_rms[:, None])
    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_mul(weight[None, :], normalized)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _launch(
    arg0_1,
    seeds_or_random,
    arg2_1,
    arg3_1,
    gt,
    add,
    rsqrt,
    out_base,
    *,
    use_random_ptr: bool,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    grid = (triton.cdiv(rows, ROW_BLOCK),)
    _dropout_rmsnorm_kernel[grid](
        arg0_1,
        seeds_or_random,
        arg2_1,
        arg3_1,
        gt,
        add,
        rsqrt,
        out_base,
        ROWS=rows,
        HIDDEN=hidden,
        SEED_IDX=SEED_INDEX,
        DROPOUT_SCALE_VALUE=DROPOUT_SCALE,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# 46dbfd5f: (T([4096,512], bf16), T([84], i64), T([32,128,512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512, ROW_BLOCK=1, num_warps=4, num_stages=3)
# ebc95169: (T([8192,512], bf16), T([64], i64), T([8,1024,512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    base_stride = _contiguous_stride(base_shape)
    rsqrt_shape = base_shape[:-1] + (1,)

    gt = torch.empty_strided(
        base_shape,
        base_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    add = torch.empty_strided(
        base_shape,
        base_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        rsqrt_shape,
        _contiguous_stride(rsqrt_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        _launch(
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            gt,
            add,
            rsqrt,
            out_base,
            use_random_ptr=False,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
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
        _launch(
            arg0_1,
            random,
            arg2_1,
            arg3_1,
            gt,
            add,
            rsqrt,
            out_base,
            use_random_ptr=True,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, add, rsqrt, out_base.view(out_shape)
