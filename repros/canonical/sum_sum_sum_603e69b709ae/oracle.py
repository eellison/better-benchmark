"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 avg_pool2d_backward plus exact-GELU-gradient fanout, including the returned singleton sigmoid, both returned full-resolution bf16 tensors, the f32 scalar reduction, the dependent per-`(N,C)` gate-gradient tensor, and both final f32 channel vectors from shared spatial partials, whereas Inductor materializes the structured pool-backward expansion, decomposed pointwise chain, scalar reduction, spatial reduction, sigmoid-derivative epilogue, and sibling channel reductions as separate generic regions over channels-last dense tensors. Inductor cannot do this today because its scheduler does not form one dtype-boundary-preserving multi-output reduction plan that shares the fixed 2x2 pool-backward producer across the scalar, per-row, and channel reductions while keeping the visible bf16 casts and side outputs. The fix is SCHEDULER_FUSION: emit a guarded NFNet structured-pool exact-GELU-gradient template that stores the visible outputs and compact reduction partials from one spatial loop, then finalizes the scalar and channel outputs directly."""

import torch
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
def _to_bf16_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _sigmoid_kernel(
    gate_ptr,
    sigmoid_out_ptr,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NC
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(sigmoid_out_ptr + offsets, sigmoid, mask=mask)


@triton.jit
def _nfnet_pool_gelu_spatial_kernel(
    pool_grad_ptr,
    x2_ptr,
    sigmoid_out_ptr,
    x4_ptr,
    scalar_ptr,
    x6_ptr,
    out1_ptr,
    out3_ptr,
    partial_row_ptr,
    partial_out1_ptr,
    partial_scalar_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    PG_S0: tl.constexpr,
    PG_S1: tl.constexpr,
    PG_S2: tl.constexpr,
    PG_S3: tl.constexpr,
    X2_S0: tl.constexpr,
    X2_S1: tl.constexpr,
    X2_S2: tl.constexpr,
    X2_S3: tl.constexpr,
    X4_S0: tl.constexpr,
    X4_S1: tl.constexpr,
    X4_S2: tl.constexpr,
    X4_S3: tl.constexpr,
    X6_S0: tl.constexpr,
    X6_S1: tl.constexpr,
    X6_S2: tl.constexpr,
    X6_S3: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    O_S2: tl.constexpr,
    O_S3: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    hw_block = tl.program_id(2)

    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_mask = c < C
    hw_mask = hw < (H * W)
    mask = hw_mask[:, None] & c_mask[None, :]

    h = hw // W
    w = hw - h * W
    ph = h // 2
    pw = w // 2

    pool_offsets = n * PG_S0 + c[None, :] * PG_S1 + ph[:, None] * PG_S2 + pw[:, None] * PG_S3
    x2_offsets = n * X2_S0 + c[None, :] * X2_S1 + h[:, None] * X2_S2 + w[:, None] * X2_S3
    x4_offsets = n * X4_S0 + c[None, :] * X4_S1 + h[:, None] * X4_S2 + w[:, None] * X4_S3
    x6_offsets = n * X6_S0 + c[None, :] * X6_S1 + h[:, None] * X6_S2 + w[:, None] * X6_S3
    out_offsets = n * O_S0 + c[None, :] * O_S1 + h[:, None] * O_S2 + w[:, None] * O_S3

    pool_grad = tl.load(pool_grad_ptr + pool_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + x2_offsets, mask=mask, other=0.0).to(tl.float32)
    x4 = tl.load(x4_ptr + x4_offsets, mask=mask, other=0.0).to(tl.float32)
    x6 = tl.load(x6_ptr + x6_offsets, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    gate_sigmoid = tl.load(
        sigmoid_out_ptr + n * C + c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    pool_expanded = _to_bf16_f32(_f32_mul(pool_grad, 0.25))
    add = _to_bf16_f32(_f32_add(x2, pool_expanded))
    mul = _to_bf16_f32(_f32_mul(add, 0.9805806756909201))
    mul1 = _to_bf16_f32(_f32_mul(mul, 1.7015043497085571))

    mul2 = _f32_mul(x4, gate_sigmoid[None, :]).to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul3 = _f32_mul(mul2.to(tl.float32), 2.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul4 = _f32_mul(mul3.to(tl.float32), scalar).to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul5 = _f32_mul(mul4.to(tl.float32), 0.2).to(tl.bfloat16, fp_downcast_rounding="rtne")
    activation = _to_bf16_f32(_f32_add(mul5.to(tl.float32), x6))

    erf_arg = _f32_mul(activation, 0.7071067811865476)
    erf = libdevice.erf(erf_arg)
    add2 = _f32_add(erf, 1.0)
    mul7 = _f32_mul(add2, 0.5)
    mul8 = _f32_mul(activation, activation)
    mul9 = _f32_mul(mul8, -0.5)
    exp = libdevice.exp(mul9)
    mul10 = _f32_mul(exp, 0.3989422804014327)
    mul11 = _f32_mul(activation, mul10)
    add3 = _f32_add(mul7, mul11)
    mul12 = _f32_mul(mul1, add3)
    out1_bf16 = mul12.to(tl.bfloat16, fp_downcast_rounding="rtne")

    mul13 = _f32_mul(out1_bf16.to(tl.float32), 0.2).to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul14 = _f32_mul(mul13.to(tl.float32), mul3.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    mul15 = _f32_mul(mul13.to(tl.float32), scalar).to(tl.bfloat16, fp_downcast_rounding="rtne")
    out3_bf16 = _f32_mul(mul15.to(tl.float32), 2.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul17 = _f32_mul(out3_bf16.to(tl.float32), x4).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out1_ptr + out_offsets, out1_bf16, mask=mask)
    tl.store(out3_ptr + out_offsets, out3_bf16, mask=mask)

    partial_base = (hw_block * N + n) * C + c
    tl.store(
        partial_row_ptr + partial_base,
        tl.sum(tl.where(mask, mul17.to(tl.float32), 0.0), axis=0),
        mask=c_mask,
    )
    tl.store(
        partial_out1_ptr + partial_base,
        tl.sum(tl.where(mask, out1_bf16.to(tl.float32), 0.0), axis=0),
        mask=c_mask,
    )

    scalar_by_c = tl.sum(tl.where(mask, mul14.to(tl.float32), 0.0), axis=0)
    scalar_partial = tl.sum(tl.where(c_mask, scalar_by_c, 0.0), axis=0)
    tl.store(
        partial_scalar_ptr + (hw_block * N + n) * NUM_C_BLOCKS + c_block,
        scalar_partial,
    )


@triton.jit
def _nfnet_finalize_nc_kernel(
    sigmoid_out_ptr,
    partial_row_ptr,
    row_out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    ROW_S0: tl.constexpr,
    ROW_S1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    c_mask = c < C
    n_mask = n < N
    mask = n_mask[:, None] & c_mask[None, :]

    spatial = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    for hb in tl.static_range(0, NUM_HW_BLOCKS):
        offsets = (hb * N + n[:, None]) * C + c[None, :]
        spatial += tl.load(partial_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid = tl.load(sigmoid_out_ptr + n[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    gate_deriv = _f32_mul(sigmoid, _f32_sub(1.0, sigmoid))
    row_bf16 = _f32_mul(
        spatial.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
        gate_deriv,
    ).to(tl.bfloat16, fp_downcast_rounding="rtne")

    row_offsets = n[:, None] * ROW_S0 + c[None, :] * ROW_S1
    tl.store(row_out_ptr + row_offsets, row_bf16, mask=mask)


@triton.jit
def _nfnet_finalize_channel_kernel(
    partial_out1_ptr,
    row_out_ptr,
    final_gate_ptr,
    final_out1_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    ROW_S0: tl.constexpr,
    ROW_S1: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C

    p = tl.arange(0, BLOCK_PARTIALS)
    partial_mask = (p[:, None] < NUM_PARTIALS) & c_mask[None, :]
    partial_offsets = p[:, None] * C + c[None, :]
    out1_sum = tl.sum(
        tl.load(partial_out1_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32),
        axis=0,
    )

    n = tl.arange(0, BLOCK_N)
    n_mask = (n[:, None] < N) & c_mask[None, :]
    row_offsets = n[:, None] * ROW_S0 + c[None, :] * ROW_S1
    row_sum = tl.sum(
        tl.load(row_out_ptr + row_offsets, mask=n_mask, other=0.0).to(tl.float32),
        axis=0,
    )

    tl.store(final_gate_ptr + c, row_sum, mask=c_mask)
    tl.store(final_out1_ptr + c, out1_sum, mask=c_mask)


@triton.jit
def _scalar_finalize_kernel(
    partial_scalar_ptr,
    scalar_out_ptr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < COUNT
    values = tl.load(partial_scalar_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(scalar_out_ptr, total)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="9c141705",
    BLOCK_HW=64,
    BLOCK_C=32,
    FINAL_BLOCK_C=8,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    pool_grad, _pool_shape, x2, gate, x4, scalar, x6 = inputs
    n = int(x2.shape[0])
    c = int(x2.shape[1])
    h = int(x2.shape[2])
    w = int(x2.shape[3])
    hw_blocks = triton.cdiv(h * w, BLOCK_HW)
    c_blocks = triton.cdiv(c, BLOCK_C)

    out_stride = tuple(int(s) for s in x2.stride())
    gate_stride = tuple(int(s) for s in gate.stride())
    sigmoid_out = torch.empty_strided(tuple(gate.shape), gate_stride, device=gate.device, dtype=torch.bfloat16)
    out1 = torch.empty_strided(tuple(x2.shape), out_stride, device=x2.device, dtype=torch.bfloat16)
    scalar_out = torch.empty((), device=x2.device, dtype=torch.float32)
    out3 = torch.empty_strided(tuple(x2.shape), out_stride, device=x2.device, dtype=torch.bfloat16)
    row_out = torch.empty_strided(tuple(gate.shape), gate_stride, device=gate.device, dtype=torch.bfloat16)
    final_gate = torch.empty_strided((c,), (1,), device=x2.device, dtype=torch.float32)
    final_out1 = torch.empty_strided((c,), (1,), device=x2.device, dtype=torch.float32)
    partial_row = torch.empty((hw_blocks, n, c), device=x2.device, dtype=torch.float32)
    partial_out1 = torch.empty((hw_blocks, n, c), device=x2.device, dtype=torch.float32)
    partial_scalar = torch.empty((hw_blocks, n, c_blocks), device=x2.device, dtype=torch.float32)

    pool_s = tuple(int(s) for s in pool_grad.stride())
    x2_s = tuple(int(s) for s in x2.stride())
    x4_s = tuple(int(s) for s in x4.stride())
    x6_s = tuple(int(s) for s in x6.stride())

    _sigmoid_kernel[(triton.cdiv(n * c, 256),)](
        gate,
        sigmoid_out,
        NC=n * c,
        BLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    _nfnet_pool_gelu_spatial_kernel[(n, c_blocks, hw_blocks)](
        pool_grad,
        x2,
        sigmoid_out,
        x4,
        scalar,
        x6,
        out1,
        out3,
        partial_row,
        partial_out1,
        partial_scalar,
        N=n,
        C=c,
        H=h,
        W=w,
        NUM_C_BLOCKS=c_blocks,
        PG_S0=pool_s[0],
        PG_S1=pool_s[1],
        PG_S2=pool_s[2],
        PG_S3=pool_s[3],
        X2_S0=x2_s[0],
        X2_S1=x2_s[1],
        X2_S2=x2_s[2],
        X2_S3=x2_s[3],
        X4_S0=x4_s[0],
        X4_S1=x4_s[1],
        X4_S2=x4_s[2],
        X4_S3=x4_s[3],
        X6_S0=x6_s[0],
        X6_S1=x6_s[1],
        X6_S2=x6_s[2],
        X6_S3=x6_s[3],
        O_S0=out_stride[0],
        O_S1=out_stride[1],
        O_S2=out_stride[2],
        O_S3=out_stride[3],
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _nfnet_finalize_nc_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        sigmoid_out,
        partial_row,
        row_out,
        N=n,
        C=c,
        NUM_HW_BLOCKS=hw_blocks,
        ROW_S0=gate_stride[0],
        ROW_S1=gate_stride[1],
        BLOCK_N=128,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _nfnet_finalize_channel_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial_out1,
        row_out,
        final_gate,
        final_out1,
        N=n,
        C=c,
        NUM_PARTIALS=hw_blocks * n,
        ROW_S0=gate_stride[0],
        ROW_S1=gate_stride[1],
        BLOCK_PARTIALS=_next_power_of_2(hw_blocks * n),
        BLOCK_N=128,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _scalar_finalize_kernel[(1,)](
        partial_scalar,
        scalar_out,
        COUNT=hw_blocks * n * c_blocks,
        BLOCK=_next_power_of_2(hw_blocks * n * c_blocks),
        num_warps=8,
        num_stages=3,
    )
    return sigmoid_out, out1, scalar_out, out3, row_out, final_gate, final_out1
