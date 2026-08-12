"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 seeded dropout-residual RMSNorm training scope in one Triton row kernel, including the flat-to-3D view, seed-index-18 Inductor dropout with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling, returned bool mask, returned fp32 residual add, fp32 mean-square reduction with eps=1e-6 rsqrt side output, affine f32 epilogue, final bf16 cast, and flattened output view, whereas Inductor lowers the stochastic producer, RMS reduction, affine store, bf16 cast, and returned side tensors through generic scheduler boundaries; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden RMSNorm lowering does not keep the seeded dropout mask producer and all visible side outputs resident across the row-statistics pass and affine epilogue while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach the RMSNorm scheduler to inline Inductor-seeded dropout and emit the returned mask, residual add, rsqrt side tensor, and bf16 view from one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_INDEX = 18
HIDDEN = 512
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


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
def _dropout_residual_rmsnorm_kernel(
    flat_ptr,
    seeds_or_random_ptr,
    residual_ptr,
    weight_ptr,
    mask_out_ptr,
    add_out_ptr,
    rsqrt_out_ptr,
    final_out_ptr,
    ROWS: tl.constexpr,
    RNG_SEED_INDEX: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = rows < ROWS
    col_mask = cols < BLOCK_H
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * BLOCK_H + cols[None, :]

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if USE_RANDOM_PTR:
        random_bf16 = tl.load(
            seeds_or_random_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(seeds_or_random_ptr + RNG_SEED_INDEX)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    threshold = tl.full((BLOCK_M, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    tl.store(mask_out_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, flat, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(tl.bfloat16)
    add_value = residual + scaled.to(tl.float32)
    tl.store(add_out_ptr + offsets, add_value, mask=mask)

    square_sum = tl.sum(tl.where(mask, add_value * add_value, 0.0), axis=1)
    inv_rms = tl.rsqrt(square_sum / BLOCK_H + 1.0e-6)
    tl.store(rsqrt_out_ptr + rows, inv_rms, mask=row_mask)

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(add_value, inv_rms[:, None])
    final = _f32_mul(weight[None, :], normalized).to(tl.bfloat16)
    tl.store(final_out_ptr + offsets, final, mask=mask)


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
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        props.multi_processor_count * blocks_per_sm,
        (int(numel) + block_size - 1) // block_size,
    )
    unroll = 4
    return (((int(numel) - 1) // (block_size * grid * unroll) + 1) * 4) * 2


def _inductor_random_for_eager_check(shape, seed, *, device):
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


def _launch(
    arg0_1,
    seeds_or_random,
    arg2_1,
    arg3_1,
    gt,
    add,
    rsqrt,
    out,
    *,
    use_random_ptr: bool,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    rows = int(arg0_1.shape[0])
    grid = (triton.cdiv(rows, BLOCK_M),)
    _dropout_residual_rmsnorm_kernel[grid](
        arg0_1,
        seeds_or_random,
        arg2_1,
        arg3_1,
        gt,
        add,
        rsqrt,
        out,
        ROWS=rows,
        RNG_SEED_INDEX=SEED_INDEX,
        USE_RANDOM_PTR=use_random_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_M=1, BLOCK_H=512, num_warps=4, num_stages=2)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_M=1, BLOCK_H=512, num_warps=4, num_stages=2)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape2)
    base_stride = _contiguous_stride(base_shape)

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
    rsqrt_shape = base_shape[:-1] + (1,)
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
            BLOCK_M=BLOCK_M,
            BLOCK_H=BLOCK_H,
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
            BLOCK_M=BLOCK_M,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, add, rsqrt, out_base.view(out_shape)
