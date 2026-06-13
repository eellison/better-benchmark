"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 seeded dropout-residual-RMSNorm training scope in one Triton row kernel, including the flat-to-`[B,S,512]` view, Inductor seed-index-56 dropout with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling, fp32 residual add, mean-square reduction, eps=1e-6 rsqrt, affine weight multiply, returned mask/add/rsqrt side outputs, and bf16 flattened view output, whereas Inductor lowers the stochastic producer, RMS mean reduction, affine epilogue, bf16 cast, and returned side tensors through generic scheduler boundaries; Inductor cannot do this today because its normalization template matcher does not canonicalize an Inductor-seeded bf16 dropout-add producer feeding RMSNorm while keeping every returned side output resident through the row reduction and epilogue; the fix is NEW_PATTERN: add a guarded dropout-residual-RMSNorm lowering that threads Inductor RNG through the row-normalization template and emits the mask, residual add, rsqrt, and bf16 affine view from one plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _finish_row(
    src,
    random,
    residual,
    weight,
    gt_ptr,
    add_ptr,
    rsqrt_ptr,
    out_ptr,
    offsets,
    rows,
    cols,
    mask,
    row_mask,
    col_mask,
    HIDDEN: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rand_bf16 = random.to(tl.bfloat16)
    dropout_p = tl.full((ROW_BLOCK, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, src, 0.0).to(tl.bfloat16)
    dropped_scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(
        tl.bfloat16
    )
    x = _f32_add(residual, dropped_scaled.to(tl.float32))
    tl.store(add_ptr + offsets, x, mask=mask)

    x_masked = tl.where(mask, x, 0.0)
    square_sum = tl.sum(_f32_mul(x_masked, x_masked), axis=1)
    mean_square = square_sum / HIDDEN
    inv_rms = libdevice.rsqrt(_f32_add(mean_square, 1.0e-6))
    tl.store(rsqrt_ptr + rows, inv_rms, mask=row_mask)

    w = tl.load(
        weight + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    normed = _f32_mul(x, inv_rms[:, None])
    out = _f32_mul(w, normed).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _dropout_residual_rmsnorm_seeded_kernel(
    flat_ptr,
    seeds_ptr,
    residual_ptr,
    weight_ptr,
    gt_ptr,
    add_ptr,
    rsqrt_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    offsets = rows[:, None] * HIDDEN + cols[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]

    src = tl.load(
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
    seed = tl.load(seeds_ptr + 56)
    random = tl.rand(seed, offsets.to(tl.uint32))

    _finish_row(
        src,
        random,
        residual,
        weight_ptr,
        gt_ptr,
        add_ptr,
        rsqrt_ptr,
        out_ptr,
        offsets,
        rows,
        cols,
        mask,
        row_mask,
        col_mask,
        HIDDEN,
        ROW_BLOCK,
        BLOCK_H,
    )


@triton.jit
def _dropout_residual_rmsnorm_random_kernel(
    flat_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    gt_ptr,
    add_ptr,
    rsqrt_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    offsets = rows[:, None] * HIDDEN + cols[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]

    src = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    random = tl.load(
        random_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    _finish_row(
        src,
        random,
        residual,
        weight_ptr,
        gt_ptr,
        add_ptr,
        rsqrt_ptr,
        out_ptr,
        offsets,
        rows,
        cols,
        mask,
        row_mask,
        col_mask,
        HIDDEN,
        ROW_BLOCK,
        BLOCK_H,
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
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


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
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
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    rsqrt_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    add = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        rsqrt_shape,
        _contiguous_stride(rsqrt_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    _dropout_residual_rmsnorm_seeded_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        gt,
        add,
        rsqrt,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if torch.cuda.is_current_stream_capturing():
        return (gt, add, rsqrt, out)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 56)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)
    _dropout_residual_rmsnorm_random_kernel[grid](
        arg0_1,
        random,
        arg2_1,
        arg3_1,
        gt,
        add,
        rsqrt,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (gt, add, rsqrt, out)
