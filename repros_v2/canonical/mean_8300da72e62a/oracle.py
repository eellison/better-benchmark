"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MT5 seeded dropout-residual-RMSNorm training scope in one Triton row kernel, including the flat-to-`[32,128,512]` view, seed-index-80 dropout with the required f32-random-to-bf16 compare boundary, returned bool mask, returned fp32 residual sum, fp32 mean-square reduction with eps=1e-6, returned rsqrt, fp32 affine multiply, bf16 output cast, and final flattened view, whereas Inductor lowers the stochastic producer, residual add, RMS reduction, affine epilogue, and visible side outputs through generic RNG/reduction/pointwise scheduling; Inductor cannot do this today because its norm-template canonicalization does not recognize an Inductor-seeded dropout-add producer feeding fixed-hidden RMSNorm while preserving all returned side outputs and bf16 rounding boundaries; the fix is NEW_PATTERN: add a guarded seeded-dropout residual RMSNorm template that threads the RNG mask through the row reduction and emits the mask, residual sum, inverse RMS, and affine output view from one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 80
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
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
    HIDDEN: tl.constexpr,
    RNG_SEED_INDEX: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    EPS_C: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
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
    )
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
        seed = tl.load(seeds_or_random_ptr + RNG_SEED_INDEX)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    threshold = tl.full((BLOCK_M, BLOCK_H), DROPOUT_P_C, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    tl.store(mask_out_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, flat, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_C).to(tl.bfloat16)
    add_value = _f32_add(residual, scaled.to(tl.float32))
    tl.store(add_out_ptr + offsets, add_value, mask=mask)

    square_sum = tl.sum(
        tl.where(mask, _f32_mul(add_value, add_value), 0.0),
        axis=1,
    )
    inv_rms = libdevice.rsqrt(_f32_add(square_sum / HIDDEN, EPS_C))
    tl.store(rsqrt_out_ptr + rows, inv_rms, mask=row_mask)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
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


@oracle_impl(
    hardware="B200",
    point="46dbfd5f",
    BLOCK_M=1,
    BLOCK_H=512,
    num_warps=4,
    num_stages=2,
)
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
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])

    mask_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    add_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt_shape = base_shape[:-1] + (1,)
    rsqrt_out = torch.empty_strided(
        rsqrt_shape,
        _contiguous_stride(rsqrt_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_base = torch.empty_strided(
        base_shape,
        base_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_residual_rmsnorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            mask_out,
            add_out,
            rsqrt_out,
            final_base,
            ROWS=rows,
            HIDDEN=hidden,
            RNG_SEED_INDEX=SEED_INDEX,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            USE_RANDOM_PTR=False,
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
        _dropout_residual_rmsnorm_kernel[grid](
            arg0_1,
            random,
            arg2_1,
            arg3_1,
            mask_out,
            add_out,
            rsqrt_out,
            final_base,
            ROWS=rows,
            HIDDEN=hidden,
            RNG_SEED_INDEX=SEED_INDEX,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            USE_RANDOM_PTR=True,
            BLOCK_M=BLOCK_M,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return mask_out, add_out, rsqrt_out, final_base.view(out_shape)
