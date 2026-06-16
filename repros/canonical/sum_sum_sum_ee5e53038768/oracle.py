"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle mirrors the compiled ConvNeXtV2 GRN/GELU multi-output schedule for the complete captured scope, including the returned f32 channel reductions, scalar zero, dense bf16 output, and final f32 channel reduction from the materialized dense output, whereas Inductor already emits a near-floor generated multi-kernel schedule for this shape; Inductor cannot do this today because the generic generated schedule is not represented as a reusable guarded fusion for this recurrent ConvNeXtV2 pattern; the fix is SCHEDULER_FUSION: add a pattern-specific GRN/GELU multi-output schedule that preserves reduction chunking, libdevice erf/tl_math.exp lowering, bf16 stores, output strides, and full return scope."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

from oracle_harness import oracle_impl


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


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
def _partial01_kernel(
    in0,
    in1,
    norm,
    denom,
    out_sum2,
    out_sum1,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 125440
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex % 2560
    x1 = xindex // 2560
    acc_sum2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    acc_sum1 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        linear_nhw = r0_index + 128 * x1
        a = tl.load(
            in0 + (x0 + 2560 * r0_index + 327680 * x1),
            xmask & r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            in1 + (x0 + 2560 * r0_index + 327680 * x1),
            xmask & r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        norm_v = tl.load(
            norm + (x0 + 2560 * (linear_nhw // 49)),
            xmask & r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        denom_v = tl.load(
            denom + (linear_nhw // 49),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        gelu = _bf16((b * 0.5) * (libdevice.erf(b * 0.7071067811865476) + 1.0))
        term = a * (gelu * (norm_v / denom_v))
        acc_sum2 += tl.where(xmask & r0_mask, term, 0.0)
        acc_sum1 += tl.where(xmask & r0_mask, a, 0.0)
    tl.store(out_sum2 + xindex, tl.sum(acc_sum2, 1)[:, None], xmask)
    tl.store(out_sum1 + xindex, tl.sum(acc_sum1, 1)[:, None], xmask)


@triton.jit
def _reduce_hw_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
):
    xnumel = 2560
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    vals = tl.load(in_ptr + (xindex + 2560 * r0_index), r0_mask & xmask, other=0.0)
    vals = tl.where(r0_mask & xmask, vals, 0.0)
    tl.store(out_ptr + xindex, tl.sum(vals, 1)[:, None], xmask)


@triton.jit
def _sum3_kernel(
    in0,
    weight,
    in1,
    out,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
):
    xnumel = 327680
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    x0 = xindex % 2560
    x1 = xindex // 2560
    a = tl.load(
        in0 + (x0 + 2560 * r0_index + 125440 * x1),
        r0_mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(weight + x0, None, eviction_policy="evict_last")
    b = tl.load(
        in1 + (x0 + 2560 * r0_index + 125440 * x1),
        r0_mask,
        other=0.0,
    ).to(tl.float32)
    gelu = _bf16((b * 0.5) * (libdevice.erf(b * 0.7071067811865476) + 1.0))
    term = (a * w) * gelu
    tl.store(out + xindex, tl.sum(tl.where(r0_mask, term, 0.0), 1)[:, None], None)


@triton.jit
def _summary_kernel(
    arg0,
    gelu_bf16,
    arg4,
    sum_a,
    sum_a_gelu,
    sum_a_weight_gelu,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    n = tl.program_id(1)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c < 2560
    gamma = tl.load(arg4 + c, mask=mask, other=0.0).to(tl.float32)

    sum_a0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_a1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_a2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_a3 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_ag0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_ag1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_ag2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_ag3 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_awg0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_awg1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_awg2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    sum_awg3 = tl.zeros([BLOCK_C], dtype=tl.float32)
    for hw_idx in tl.static_range(0, 49):
        h = hw_idx // 7
        w = hw_idx - h * 7
        offs = n * 125440 + c + h * 17920 + w * 2560
        a = tl.load(arg0 + offs, mask=mask, other=0.0).to(tl.float32)
        gelu = tl.load(gelu_bf16 + offs, mask=mask, other=0.0).to(tl.float32)
        ag = tl.where(mask, a * gelu, 0.0)
        a = tl.where(mask, a, 0.0)
        awg = tl.where(mask, _f32_mul(_f32_mul(a, gamma), gelu), 0.0)
        if hw_idx % 4 == 0:
            sum_a0 = _f32_add(sum_a0, a)
            sum_ag0 = _f32_add(sum_ag0, ag)
            sum_awg0 = _f32_add(sum_awg0, awg)
        elif hw_idx % 4 == 1:
            sum_a1 = _f32_add(sum_a1, a)
            sum_ag1 = _f32_add(sum_ag1, ag)
            sum_awg1 = _f32_add(sum_awg1, awg)
        elif hw_idx % 4 == 2:
            sum_a2 = _f32_add(sum_a2, a)
            sum_ag2 = _f32_add(sum_ag2, ag)
            sum_awg2 = _f32_add(sum_awg2, awg)
        else:
            sum_a3 = _f32_add(sum_a3, a)
            sum_ag3 = _f32_add(sum_ag3, ag)
            sum_awg3 = _f32_add(sum_awg3, awg)

    sum_a_val = _f32_add(_f32_add(_f32_add(sum_a0, sum_a1), sum_a2), sum_a3)
    sum_a_gelu_val = _f32_add(_f32_add(_f32_add(sum_ag0, sum_ag1), sum_ag2), sum_ag3)
    sum_a_weight_gelu_val = _f32_add(
        _f32_add(_f32_add(sum_awg0, sum_awg1), sum_awg2), sum_awg3
    )

    out = n * 2560 + c
    tl.store(sum_a + out, sum_a_val, mask=mask)
    tl.store(sum_a_gelu + out, sum_a_gelu_val, mask=mask)
    tl.store(sum_a_weight_gelu + out, sum_a_weight_gelu_val, mask=mask)


@triton.jit
def _row_correction_kernel(
    sum3,
    norm,
    denom,
    out,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 128
    r0_numel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    denom_v = tl.load(denom + xindex, xmask, eviction_policy="evict_last")
    acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        s3 = tl.load(
            sum3 + (r0_index + 2560 * xindex),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        )
        norm_v = tl.load(
            norm + (r0_index + 2560 * xindex),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        )
        term = (-s3) * ((norm_v / denom_v) / denom_v)
        acc += tl.where(r0_mask & xmask, term, 0.0)
    tl.store(out + xindex, tl.sum(acc, 1)[:, None], xmask)


@triton.jit
def _full_zero_kernel(out, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.arange(0, XBLOCK)
    tl.store(out + (tl.full([XBLOCK], 0, tl.int32)), 0.0, None)


@triton.jit
def _dense_kernel(
    out,
    in0,
    weight,
    norm,
    denom,
    sum3,
    row_corr,
    zero,
    in1,
    xnumel,
    XBLOCK: tl.constexpr,
):
    xnumel = 16056320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x0 = xindex % 2560
    x2 = xindex // 125440
    a = tl.load(in0 + xindex, None).to(tl.float32)
    w = tl.load(weight + x0, None, eviction_policy="evict_last")
    norm_v = tl.load(norm + (x0 + 2560 * x2), None, eviction_policy="evict_last")
    denom_v = tl.load(denom + x2, None, eviction_policy="evict_last")
    s3 = tl.load(sum3 + (x0 + 2560 * x2), None, eviction_policy="evict_last")
    row_v = tl.load(row_corr + x2, None, eviction_policy="evict_last")
    zero_v = tl.load(zero + 0)
    b = tl.load(in1 + xindex, None).to(tl.float32)

    erf_term = _f32_add(libdevice.erf(_f32_mul(b, 0.7071067811865476)), 1.0)
    gelu = _f32_mul(_f32_mul(b, 0.5), erf_term).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)

    base = a.to(tl.float32)
    div = _f32_div(norm_v, denom_v)
    add1_delta = _f32_mul(_f32_mul(base, w), div).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    add1 = (base.to(tl.bfloat16, fp_downcast_rounding="rtne") + add1_delta).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    add2 = _f32_add(_f32_div(s3, denom_v), _f32_mul(row_v, 0.000390625))
    where = tl.where(norm_v == 0.0, tl.broadcast_to(zero_v, [XBLOCK]), _f32_div(gelu, norm_v))
    add3_delta = _f32_mul(add2, where).to(tl.bfloat16, fp_downcast_rounding="rtne")
    add3 = (add1 + add3_delta).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    exp_arg = _f32_mul(_f32_mul(b, b), -0.5)
    grad = _f32_add(
        _f32_mul(erf_term, 0.5),
        _f32_mul(b, _f32_mul(tl_math.exp(exp_arg), 0.3989422804014327)),
    )
    result = _f32_mul(add3, grad).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out + xindex, result, None)


@triton.jit
def _output4_stage_kernel(
    dense,
    partial,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 125440
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex % 2560
    x1 = xindex // 2560
    acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        val = tl.load(
            dense + (x0 + 2560 * r0_index + 327680 * x1),
            xmask & r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        acc += tl.where(xmask & r0_mask, val, 0.0)
    tl.store(partial + xindex, tl.sum(acc, 1)[:, None], xmask)


@triton.jit
def _finalize01_kernel(
    norm,
    denom,
    sum_a,
    sum_a_gelu,
    out0,
    out1,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_c = tl.program_id(0)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    mask = (c[None, :] < 2560) & (n[:, None] < 128)
    offs = n[:, None] * 2560 + c[None, :]
    p = tl.load(norm + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(denom + n, mask=n < 128, other=1.0).to(tl.float32)
    q = tl.load(sum_a_gelu + offs, mask=mask, other=0.0).to(tl.float32)
    a_sum = tl.load(sum_a + offs, mask=mask, other=0.0).to(tl.float32)
    out0_val = tl.sum(tl.where(mask, _f32_mul(q, _f32_div(p, s[:, None])), 0.0), axis=0)
    out1_val = tl.sum(tl.where(mask, a_sum, 0.0), axis=0)
    valid = c < 2560
    tl.store(out0 + c, out0_val, mask=valid)
    tl.store(out1 + c, out1_val, mask=valid)


@oracle_impl(hardware="B200", point="743aa381")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2 = inputs
    del _shape0, _shape1, _shape2
    device = arg0_1.device

    sum_a = torch.empty_strided((128, 2560), (2560, 1), device=device, dtype=torch.float32)
    sum_a_gelu = torch.empty_strided(
        (128, 2560), (2560, 1), device=device, dtype=torch.float32
    )
    sum3 = torch.empty_strided((128, 2560), (2560, 1), device=device, dtype=torch.float32)
    gelu_bf16 = torch.nn.functional.gelu(arg1_1, approximate="none")
    _summary_kernel[(triton.cdiv(2560, 256), 128)](
        arg0_1,
        gelu_bf16,
        arg4_1,
        sum_a,
        sum_a_gelu,
        sum3,
        BLOCK_C=256,
        num_warps=4,
    )
    div = arg2_1 / arg3_1
    row_corr = ((-sum3) * ((div / arg3_1).reshape(128, 2560))).sum(
        dim=1, dtype=torch.float32
    ).contiguous()

    out0 = torch.empty_strided((2560,), (1,), device=device, dtype=torch.float32)
    out1 = torch.empty_strided((2560,), (1,), device=device, dtype=torch.float32)
    _finalize01_kernel[(triton.cdiv(2560, 64),)](
        arg2_1,
        arg3_1,
        sum_a,
        sum_a_gelu,
        out0,
        out1,
        BLOCK_C=64,
        BLOCK_N=128,
        num_warps=8,
        num_stages=1,
    )

    out2 = torch.empty_strided((), (), device=device, dtype=torch.float32)
    _full_zero_kernel[(1,)](out2, 1, XBLOCK=1, num_warps=1, num_stages=1)

    out3 = torch.empty_strided(
        (128, 2560, 7, 7),
        (125440, 1, 17920, 2560),
        device=device,
        dtype=torch.bfloat16,
    )
    _dense_kernel[(triton.cdiv(16056320, 1024),)](
        out3,
        arg0_1,
        arg4_1,
        arg2_1,
        arg3_1,
        sum3,
        row_corr,
        out2,
        arg1_1,
        16056320,
        XBLOCK=1024,
        num_warps=4,
        num_stages=1,
    )

    out4_partial = torch.empty_strided(
        (2560, 49), (1, 2560), device=device, dtype=torch.float32
    )
    _output4_stage_kernel[(triton.cdiv(125440, 64),)](
        out3,
        out4_partial,
        125440,
        128,
        XBLOCK=64,
        R0_BLOCK=8,
        num_warps=4,
        num_stages=1,
    )
    out4 = torch.empty_strided((2560,), (1,), device=device, dtype=torch.float32)
    _reduce_hw_kernel[(triton.cdiv(2560, 32),)](
        out4_partial,
        out4,
        2560,
        49,
        XBLOCK=32,
        num_warps=8,
        num_stages=1,
    )

    return (
        out0,
        out1,
        out2,
        out3,
        out4,
    )
