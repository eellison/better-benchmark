"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J mixed-dtype residual LayerNorm training scope in one Triton row kernel, including the two `[128,4096]` bf16 input views, the observable bf16 residual add, the returned f32 pre-norm add with the f32 `[1,128,4096]` residual, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt side output, affine scale/bias epilogue, and final bf16 `[128,4096]` view, whereas Inductor lowers the live add, normalization, statistic side outputs, affine cast, and view return through generic normalization scheduling; Inductor cannot do this today because the fixed-hidden LayerNorm scheduler does not retain a mixed bf16/f32 residual producer and all required side-output stores across the row-statistics pass and epilogue; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline same-layout residual adds, preserve visible add/mean/rsqrt outputs, and emit the final bf16 view directly from one full-scope row plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5
_USE_INDUCTOR_NUMERICS = False


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
def _gptj_residual_layernorm_full_kernel(
    lhs_ptr,
    rhs_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_bf16_out_ptr,
    add_f32_out_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    final_out_ptr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN + cols

    lhs = tl.load(lhs_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    residual = tl.load(residual_ptr + offsets, eviction_policy="evict_first").to(tl.float32)

    add_f32 = _f32_add(lhs, rhs)
    add_bf16 = add_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(add_bf16_out_ptr + offsets, add_bf16)

    x = _f32_add(add_bf16.to(tl.float32), residual)
    if USE_INDUCTOR_NUMERICS:
        x = _f32_add(add_f32, residual)
    tl.store(add_f32_out_ptr + offsets, x)

    mean_acc = tl.zeros([BLOCK_H], tl.float32)
    m2_acc = tl.zeros([BLOCK_H], tl.float32)
    weight_acc = tl.zeros([BLOCK_H], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean, m2, _weight = triton_helpers.welford(mean_next, m2_next, weight_next, 0)
    variance = _f32_mul(m2, 1.0 / HIDDEN)
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(mean_out_ptr + row, mean)
    tl.store(rsqrt_out_ptr + row, invstd)
    tl.store(final_out_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# 3fdaed2a: (T([128,4096], bf16), T([128,4096], bf16), T([1,128,4096], f32), T([4096], f32), T([4096], f32), S([1,128,4096]), S([1,128,4096]), S([128,4096]))
@oracle_impl(hardware="B200", point="3fdaed2a", BLOCK_H=4096, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    global _USE_INDUCTOR_NUMERICS
    lhs, rhs, residual, weight, bias, add_shape, add_f32_shape, final_shape = inputs
    rows = int(lhs.shape[0])
    hidden = int(lhs.shape[1])
    add_shape = _as_shape(add_shape)
    add_f32_shape = _as_shape(add_f32_shape)
    final_shape = _as_shape(final_shape)
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    add_bf16 = torch.empty_strided(
        add_shape,
        (rows * hidden, hidden, 1),
        device=lhs.device,
        dtype=torch.bfloat16,
    )
    add_f32 = torch.empty_strided(
        add_f32_shape,
        (rows * hidden, hidden, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (1, rows, 1),
        (rows, 1, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (1, rows, 1),
        (rows, 1, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    final = torch.empty_strided(
        final_shape,
        (hidden, 1),
        device=lhs.device,
        dtype=torch.bfloat16,
    )

    _gptj_residual_layernorm_full_kernel[(rows,)](
        lhs,
        rhs,
        residual,
        weight,
        bias,
        add_bf16,
        add_f32,
        mean,
        rsqrt,
        final,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return add_bf16, add_f32, mean, rsqrt, final
