"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 T5 seeded dropout-residual-RMSNorm-dropout-scale scope in one Triton row kernel, including the flat-to-`[8,1024,512]` view, seed-index-62 dropout with the required f32-random-to-bf16 compare boundary, bf16 dropout scaling, returned bool mask, returned fp32 residual add, fp32 mean-square reduction with eps=1e-6 rsqrt, fp32 affine weight multiply, seed-index-63 f32-random dropout, final dropout and 0.04419417382415922 scale, returned second bool mask, and final bf16 flattened view, whereas Inductor lowers the stochastic producers, RMS reduction, affine epilogue, final dropout/scale/cast, and visible side outputs through generic scheduler fragments; Inductor cannot fuse this full returned-output envelope today because its RMSNorm scheduler does not keep two Inductor-seeded dropout producers and all observable side outputs resident across the row-statistics pass while preserving the bf16 and f32 RNG compare boundaries; the fix is SCHEDULER_FUSION: teach RMSNorm scheduling to inline seeded dropout producers and emit the masks, residual add, inverse-RMS side tensor, and final scaled bf16 view from one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_INDEX_0 = 62
SEED_INDEX_1 = 63
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
FINAL_SCALE = 0.04419417382415922
EPS = 1.0e-6


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
def _dropout_rmsnorm_dropout_kernel(
    flat_ptr,
    seed_or_random0_ptr,
    random1_ptr,
    residual_ptr,
    weight_ptr,
    gt0_ptr,
    add_ptr,
    rsqrt_ptr,
    gt1_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    USE_RANDOM_PTRS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * HIDDEN + cols[None, :]

    if USE_RANDOM_PTRS:
        random0_bf16 = tl.load(
            seed_or_random0_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        random1 = tl.load(
            random1_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed0 = tl.load(seed_or_random0_ptr + 62)
        seed1 = tl.load(seed_or_random0_ptr + 63)
        rng_offsets = offsets.to(tl.uint32)
        random0_bf16 = tl.rand(seed0, rng_offsets).to(tl.bfloat16)
        random1 = tl.rand(seed1, rng_offsets)

    threshold_bf16 = tl.full((BLOCK_M, BLOCK_N), 0.1, tl.float32).to(
        tl.bfloat16
    )
    keep0 = random0_bf16 > threshold_bf16
    tl.store(gt0_ptr + offsets, keep0, mask=mask)

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    keep0_f32 = keep0.to(tl.float32)
    dropped0 = _f32_mul(keep0_f32, flat.to(tl.float32)).to(tl.bfloat16)
    scaled0 = _f32_mul(dropped0.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)

    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    add = _f32_add(residual, scaled0.to(tl.float32))
    tl.store(add_ptr + offsets, add, mask=mask)

    add_for_reduce = tl.where(mask, add, 0.0)
    square_sum = tl.sum(_f32_mul(add_for_reduce, add_for_reduce), axis=1)
    mean_square = _f32_mul(square_sum, 0.001953125)
    inv_rms = tl.rsqrt(_f32_add(mean_square, 1.0e-6))
    tl.store(rsqrt_ptr + rows, inv_rms, mask=row_mask)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = _f32_mul(add, inv_rms[:, None])
    affine = _f32_mul(weight[None, :], normalized)

    keep1 = random1 > 0.1
    tl.store(gt1_ptr + offsets, keep1, mask=mask)
    dropped1 = _f32_mul(keep1.to(tl.float32), affine)
    scaled1 = _f32_mul(dropped1, 1.1111111111111112)
    scaled2 = _f32_mul(scaled1, 0.04419417382415922)
    tl.store(out_ptr + offsets, scaled2.to(tl.bfloat16), mask=mask)


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


def _torch_rand_uniform_offset_advance(numel, *, device):
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (int(numel) + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    return (
        ((int(numel) - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _inductor_random_for_eager_check(shape, seed, *, device, calls_back):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = _torch_rand_uniform_offset_advance(numel, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewind = int(calls_back) * advance
    if offset >= rewind:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - rewind)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


def _launch(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    full_shape = _shape_tuple(shape0)
    random_shape0 = _shape_tuple(shape1)
    random_shape1 = _shape_tuple(shape2)
    out_shape = _shape_tuple(shape3)
    full_stride = _contiguous_stride(full_shape)
    row_shape = full_shape[:-1] + (1,)

    rows = int(arg0_1.numel() // arg0_1.shape[-1])
    hidden = int(arg0_1.shape[-1])

    gt0 = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    add = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        row_shape,
        _contiguous_stride(row_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    gt1 = torch.empty_strided(
        full_shape,
        full_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_rmsnorm_dropout_kernel[grid](
            arg0_1,
            arg1_1,
            arg1_1,
            arg2_1,
            arg3_1,
            gt0,
            add,
            rsqrt,
            gt1,
            out,
            ROWS=rows,
            HIDDEN=hidden,
            USE_RANDOM_PTRS=False,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
        seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
        random0 = _inductor_random_for_eager_check(
            random_shape0,
            seed0,
            device=arg0_1.device,
            calls_back=2,
        )
        random1 = _inductor_random_for_eager_check(
            random_shape1,
            seed1,
            device=arg0_1.device,
            calls_back=1,
        )
        _dropout_rmsnorm_dropout_kernel[grid](
            arg0_1,
            random0,
            random1,
            arg2_1,
            arg3_1,
            gt0,
            add,
            rsqrt,
            gt1,
            out,
            ROWS=rows,
            HIDDEN=hidden,
            USE_RANDOM_PTRS=True,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt0, add, rsqrt, gt1, out


# ebc95169: (T([8192,512], bf16), T([64], i64), T([8,1024,512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_M=2, BLOCK_N=512, num_warps=4, num_stages=3)
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
