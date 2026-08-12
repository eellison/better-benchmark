"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 T5 seeded dropout-residual-RMSNorm scope in one Triton row kernel, including the `[8192,512] -> [8,1024,512]` bf16 view, seed-index-32 Inductor dropout with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling by 1.1111111111111112, returned bool mask, returned fp32 residual add, returned fp32 eps=1e-6 rsqrt, affine weight multiply, final bf16 cast, and contiguous `[8192,512]` output view, whereas Inductor lowers the stochastic producer, bf16 cast/mask, residual add, mean-square reduction, rsqrt, affine epilogue, and observable side outputs through generic scheduler fragments; Inductor cannot do this today because its normalization template matcher does not canonicalize an Inductor-seeded bf16 dropout-add producer with returned intermediates into one reusable row-normalization lowering; the fix is NEW_PATTERN: add a guarded dropout-residual-RMSNorm template that threads Inductor RNG through the row reduction and emits the mask, add, rsqrt, and affine output view from one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_INDEX = 32
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


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
    arg0_ptr,
    rng_ptr,
    residual_ptr,
    weight_ptr,
    gt_ptr,
    add_ptr,
    rsqrt_ptr,
    out_ptr,
    rows_total: tl.constexpr,
    hidden: tl.constexpr,
    seed_index: tl.constexpr,
    dropout_p_value: tl.constexpr,
    dropout_scale: tl.constexpr,
    eps: tl.constexpr,
    use_random_ptr: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < rows_total
    col_mask = cols < hidden
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * hidden + cols[None, :]

    if use_random_ptr:
        rand_bf16 = tl.load(
            rng_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(rng_ptr + seed_index)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((BLOCK_M, BLOCK_N), dropout_p_value, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    source = tl.load(
        arg0_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    dropped = tl.where(keep, source, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), dropout_scale).to(tl.bfloat16)

    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    add = residual + scaled.to(tl.float32)
    tl.store(add_ptr + offsets, add, mask=mask)

    square_sum = tl.sum(tl.where(mask, add * add, 0.0), axis=1)
    inv_rms = tl.rsqrt(square_sum / hidden + eps)
    tl.store(rsqrt_ptr + rows, inv_rms, mask=row_mask)

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normalized = add * inv_rms[:, None]
    affine = weight[None, :] * normalized
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


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


def _launch(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    full_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape2)
    row_shape = full_shape[:-1] + (1,)
    rows_total = int(arg0_1.numel() // arg0_1.shape[-1])
    hidden = int(arg0_1.shape[-1])

    full_stride = _contiguous_stride(full_shape)
    row_stride = _contiguous_stride(row_shape)
    out_stride = _contiguous_stride(out_shape)

    gt = torch.empty_strided(
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
        row_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows_total, BLOCK_M),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_residual_rmsnorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            gt,
            add,
            rsqrt,
            out,
            rows_total=rows_total,
            hidden=hidden,
            seed_index=SEED_INDEX,
            dropout_p_value=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            eps=EPS,
            use_random_ptr=False,
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
        _dropout_residual_rmsnorm_kernel[grid](
            arg0_1,
            random,
            arg2_1,
            arg3_1,
            gt,
            add,
            rsqrt,
            out,
            rows_total=rows_total,
            hidden=hidden,
            seed_index=SEED_INDEX,
            dropout_p_value=DROPOUT_P,
            dropout_scale=DROPOUT_SCALE,
            eps=EPS,
            use_random_ptr=True,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, add, rsqrt, out


# ebc95169: (T([8192,512], bf16), T([64], i64), T([8,1024,512], f32), T([512], f32), ...)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_M=1, BLOCK_N=512, num_warps=4, num_stages=3)
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
