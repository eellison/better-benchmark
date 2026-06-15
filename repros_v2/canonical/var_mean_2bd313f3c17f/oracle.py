"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Blenderbot bf16 residual-add LayerNorm alias scope in one Triton row kernel, including the `[2048,2560]` input view as `[16,128,2560]`, the bf16 residual-add rounding before fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and all 48 aliased `[2048,2560]` view returns from one output buffer, whereas Inductor lowers the decomposed add/cast/var_mean/affine/cast graph through generic normalization scheduling plus repeated alias-view handling; Inductor cannot do this today because the normalization scheduler does not form a fixed-hidden residual-add LayerNorm plan that preserves bf16 rounding boundaries and a large multi-output alias contract as one specialized full-scope row kernel; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to inline same-layout residual producers, keep row statistics resident, and emit repeated view aliases from the single normalized output storage."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 16
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 2560
EPS = 1.0e-5


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
def _residual_layernorm_aliases_h2560_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    EPSILON: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_mask = rows < ROWS_
    cols0 = tl.arange(0, 2048)
    cols1 = tl.arange(0, 512) + 2048
    offsets0 = rows[:, None] * HIDDEN_ + cols0[None, :]
    offsets1 = rows[:, None] * HIDDEN_ + cols1[None, :]

    flat0 = tl.load(flat_ptr + offsets0, mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )
    flat1 = tl.load(flat_ptr + offsets1, mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )
    residual0 = tl.load(
        residual_ptr + offsets0, mask=row_mask[:, None], other=0.0
    ).to(tl.float32)
    residual1 = tl.load(
        residual_ptr + offsets1, mask=row_mask[:, None], other=0.0
    ).to(tl.float32)

    added0 = _f32_add(residual0, flat0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    added1 = _f32_add(residual1, flat1).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    x0 = added0.to(tl.float32)
    x1 = added1.to(tl.float32)

    mean = _f32_mul(
        _f32_add(tl.sum(x0, axis=1), tl.sum(x1, axis=1)),
        1.0 / 2560.0,
    )
    centered0 = _f32_sub(x0, mean[:, None])
    centered1 = _f32_sub(x1, mean[:, None])
    variance = _f32_mul(
        _f32_add(
            tl.sum(_f32_mul(centered0, centered0), axis=1),
            tl.sum(_f32_mul(centered1, centered1), axis=1),
        ),
        1.0 / 2560.0,
    )
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    weight0 = tl.load(weight_ptr + cols0).to(tl.float32)
    weight1 = tl.load(weight_ptr + cols1).to(tl.float32)
    bias0 = tl.load(bias_ptr + cols0).to(tl.float32)
    bias1 = tl.load(bias_ptr + cols1).to(tl.float32)

    norm0 = _f32_mul(centered0, invstd[:, None])
    norm1 = _f32_mul(centered1, invstd[:, None])
    affine0 = _f32_add(_f32_mul(norm0, weight0[None, :]), bias0[None, :])
    affine1 = _f32_add(_f32_mul(norm1, weight1[None, :]), bias1[None, :])
    out0 = affine0.to(tl.bfloat16, fp_downcast_rounding="rtne")
    out1 = affine1.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets0, out0, mask=row_mask[:, None])
    tl.store(out_ptr + offsets1, out1, mask=row_mask[:, None])


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="7f824027",
    ROW_BLOCK=2,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(inputs, *, ROW_BLOCK: int, num_warps: int, num_stages: int):
    flat, residual, weight, bias, _input_shape, *output_shapes = inputs
    out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=flat.device,
        dtype=torch.bfloat16,
    )

    _residual_layernorm_aliases_h2560_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        flat,
        residual,
        weight,
        bias,
        out,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        EPSILON=EPS,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (out, *(out.view(_shape_tuple(shape)) for shape in output_shapes))
