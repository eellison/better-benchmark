"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GoogleFnet fp32 seeded-dropout residual LayerNorm training scope in one Triton row kernel, including the flat-to-`[32,512,768]` view, seed-index-8 Inductor RNG with the fp32 `gt(0.1)` dropout comparison, fp32 dropout scaling, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, returned bool mask, normalized f32 tensor, affine f32 tensor, complex64 affine conversion with zero imaginary lane, and `rsqrt / 768` side output, whereas Inductor lowers the stochastic producer, normalization reduction, affine epilogue, complex conversion, and sibling side-output stores through generic normalization-template fragments; Inductor cannot do this today because its fixed-hidden normalization scheduler does not keep the input-seeded dropout producer and all returned f32/complex side outputs resident across the row-statistics pass and affine epilogue while preserving the fp32 RNG and arithmetic boundaries; the fix is SCHEDULER_FUSION: teach the normalization scheduler to fuse seeded dropout, residual add, var_mean/rsqrt, affine and complex-conversion stores, and inverse-std side-output stores into one full-scope row schedule."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 8
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


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
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dropout_residual_layernorm_kernel(
    x_ptr,
    random_or_seed_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    norm_ptr,
    affine_ptr,
    complex_real_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIDDEN_F: tl.constexpr,
    SEED_IDX: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_SEEDED_RNG: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    valid = (row_offsets[:, None] < ROWS) & (cols < HIDDEN)
    offsets = rows * HIDDEN + cols

    x = tl.load(
        x_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_SEEDED_RNG:
        seed = tl.load(random_or_seed_ptr + SEED_IDX)
        random = tl.rand(seed, offsets.to(tl.uint32))
    else:
        random = tl.load(
            random_or_seed_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

    keep = random > 0.1
    tl.store(mask_ptr + offsets, keep, mask=valid)

    dropped = tl.where(keep, x, 0.0)
    dropped_scaled = _f32_mul(dropped, DROPOUT_SCALE_C)
    layernorm_input = _f32_add(dropped_scaled, residual)

    reduce_input = tl.where(valid, layernorm_input, 0.0)
    mean_1d = tl.sum(reduce_input, axis=1) / HIDDEN
    centered = _f32_sub(layernorm_input, mean_1d[:, None])
    centered_masked = tl.where(valid, centered, 0.0)
    variance_1d = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, EPS_C))
    normalized = _f32_mul(centered, invstd_1d[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    complex_offsets = offsets * 2
    tl.store(norm_ptr + offsets, normalized, mask=valid)
    tl.store(affine_ptr + offsets, affine, mask=valid)
    tl.store(complex_real_ptr + complex_offsets, affine, mask=valid)
    tl.store(complex_real_ptr + complex_offsets + 1, 0.0, mask=valid)
    tl.store(div_ptr + row_offsets, _f32_div(invstd_1d, HIDDEN_F), mask=row_offsets < ROWS)


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


def _exact_complex_for_eager_check(arg0_1, arg2_1, arg3_1, arg4_1, shape0, random):
    view = torch.ops.aten.view.default(arg0_1, shape0)
    gt = torch.ops.aten.gt.Scalar(random, 0.1)
    mul = torch.ops.aten.mul.Tensor(gt, view)
    mul_1 = torch.ops.aten.mul.Tensor(mul, DROPOUT_SCALE)
    add = torch.ops.aten.add.Tensor(mul_1, arg2_1)
    variance, mean = torch.ops.aten.var_mean.correction(
        add, [2], correction=0, keepdim=True
    )
    rsqrt = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(add, mean), rsqrt
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, arg3_1), arg4_1
    )
    return torch.ops.prims.convert_element_type.default(affine, torch.complex64)


# 3a80a44f: (T([16384,768], f32), T([13], i64), T([32,512,768], f32), T([768], f32), T([768], f32), ...)
@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.float32)
    complex_out = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.complex64)
    complex_real = torch.view_as_real(complex_out)
    div = torch.empty(div_shape, device=arg0_1.device, dtype=torch.float32)

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            mask,
            normalized,
            affine,
            complex_real,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            HIDDEN_F=float(hidden),
            SEED_IDX=SEED_INDEX,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            mask,
            normalized,
            affine,
            complex_real,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            HIDDEN_F=float(hidden),
            SEED_IDX=SEED_INDEX,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=arg0_1.device,
        )
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            random,
            arg2_1,
            arg3_1,
            arg4_1,
            mask,
            normalized,
            affine,
            complex_real,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            HIDDEN_F=float(hidden),
            SEED_IDX=SEED_INDEX,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        complex_out = _exact_complex_for_eager_check(
            arg0_1,
            arg2_1,
            arg3_1,
            arg4_1,
            norm_shape,
            random,
        )

    return mask, normalized, affine, complex_out, div
